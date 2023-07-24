"""All the following is for sqlite3 databases."""

# MIT License
#
# Copyright (c) 2023 Yann Dubois
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import logging
import random
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml_swissknife.types import PathOrIOBase

try:
    import sqlalchemy as sa
except ImportError:
    # we only use SQLalchemy for deleting and updating rows (not adding)
    pass

logging.basicConfig(level=logging.INFO)


def delete_rows_from_db(database, table_name, columns_to_select_on, df):
    """Delete rows from a table in a SQLite database based on the values of a dataframe."""
    engine = sa.create_engine(f"sqlite:///{database}")
    table = sa.Table(table_name, sa.MetaData(), autoload_with=engine)

    # Build the WHERE clause of your DELETE statement from rows in the dataframe.
    # Equivalence in SQL:
    #   WHERE (Year = <Year from row 1 of df> AND Month = <Month from row 1 of df>)
    #      OR (Year = <Year from row 2 of df> AND Month = <Month from row 2 of df>)
    #      ...
    cond = df.apply(
        lambda row: sa.and_(*[table.c[k] == row[k] for k in columns_to_select_on]),
        axis=1,
    )
    cond = sa.or_(*cond)

    # Define and execute the DELETE
    delete = table.delete().where(cond)
    with engine.connect() as conn:
        conn.execute(delete)
        conn.commit()


def prepare_to_add_to_db_sql(
    df_to_add: pd.DataFrame,
    database: PathOrIOBase,
    sql_already_annotated: str,
    is_keep_all_columns_from_db: bool = True,
    is_check_unique_primary_key: bool = False,
    table_name: Optional[str] = None,
) -> pd.DataFrame:
    """Prepare a dataframe to be added to a table in a SQLite database by removing rows already in the database.

    Parameters
    ----------
    df_to_add : pd.DataFrame
        The dataframe to add to the database.

    database : PathOrIOBase
        The database to add to.

    sql_already_annotated : str
        The SQL query to get the dataframe already in the database.

    is_keep_all_columns_from_db : bool, optional
        Whether to return all columns in DB (or only the columns in the dataframe to add).

    is_check_unique_primary_key : bool, optional
        Raise an error if you are trying to add rows with primary keys that already exist in the database but have
        different values for non-primary keys. If True needs `table_name`

    table_name : str, optional
        The name of the table in the database. Only needed if `is_check_unique_primary_key` is True.
    """
    df_db = sql_to_df(
        database=database,
        sql=sql_already_annotated,
    )
    columns = [c for c in df_db.columns if c in df_to_add.columns]

    if not is_keep_all_columns_from_db:
        df_db = df_db[columns]

    df_all = pd.concat([df_db, df_to_add[columns]]).drop_duplicates()

    if is_check_unique_primary_key:
        assert table_name is not None
        primary_keys = (
            sql_to_df(database=database, sql=f"PRAGMA table_info('{table_name}')").query("pk > 0").name.to_list()
        )
        assert len(primary_keys) > 0, f"Table {table_name} has no primary key"

        # you previously removed exact duplicates => if there are duplicates based on primary keys it means that there
        # are rows that you are trying to add which have the same primary key as a row already in the database but
        # different non-primary keys => raise an error
        is_primary_key_duplicates = df_all.duplicated(subset=primary_keys)
        if is_primary_key_duplicates.any():
            n_duplicates = is_primary_key_duplicates.sum()
            grouped = df_all[is_primary_key_duplicates].groupby(primary_keys)
            example_primary_key_duplicates = df_all.groupby(primary_keys).get_group(list(grouped.groups.keys())[0])
            raise ValueError(
                f"Trying to add {n_duplicates} rows with primary keys {primary_keys} that already exist in the database "
                f"but have different values for non-primary keys. Example:\n {example_primary_key_duplicates}"
            )

    # remove columns that are already in DB
    df_delta = get_delta_df(df_all, df_db)
    return df_delta


# depreciate once merged
def prepare_to_add_to_db(df_to_add, database, table_name, is_subset_columns=False) -> pd.DataFrame:
    """Prepare a dataframe to be added to a table in a SQLite database. by removing rows already in the database.
    and columns not in the database."""
    df_db = sql_to_df(sql=f"SELECT * FROM {table_name}", database=database)
    columns = [c for c in df_db.columns if c in df_to_add.columns]
    if is_subset_columns:
        df_db = df_db[columns]
    df_all = pd.concat([df_db, df_to_add[columns]]).drop_duplicates()
    df_delta = get_delta_df(df_all, df_db)
    return df_delta


def get_delta_df(df_all, df_subset) -> pd.DataFrame:
    """return the complement of df_subset"""
    columns = list(df_all.columns)
    df_ind = df_all.merge(df_subset.drop_duplicates(), on=columns, how="left", indicator=True)
    return df_ind.query("_merge == 'left_only' ")[columns]


def sql_to_df(database, sql, table=None, is_enforce_numeric=False, **kwargs) -> pd.DataFrame:
    """
    Return a dataframe from a SQL query.
    If `is_enforce_numeric` will ensure that the dataframe's columns are numeric when the table columns are also.
    """
    with create_connection(database) as conn:
        df = pd.read_sql(sql, conn, **kwargs)

        if is_enforce_numeric:
            assert table is not None, "If `is_type_enforce` is True, `table` must be specified."
            table_info = pd.read_sql(sql=f"PRAGMA table_info({table})", con=conn)

    if is_enforce_numeric:
        cols_to_numeric = [r["name"] for _, r in table_info.iterrows() if r["type"] in ["INTEGER", "FLOAT"]]
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = df[c].apply(pd.to_numeric, errors="coerce")

    return df


def get_values_from_keys(database, table, df, is_rm_duplicates=False):
    """Given a dataframe containing the primary keys of a table, will return the corresponding rows"""
    keys = ", ".join(df.columns)
    list_of_tuples = list(zip(*[df[c] for c in df.columns]))

    list_of_placeholders = [tuple("?" for _ in range(len(df.columns)))] * len(list_of_tuples)
    placeholders = str(list_of_placeholders)[1:-1].replace("'?'", "?")

    flattened_values = [item for sublist in list_of_tuples for item in sublist]

    try:
        # this is much faster an memory efficient but sometimes has some formatting issues
        # Example sql query is built as follows:
        #   SELECT * FROM likert_annotations WHERE (input_id, output) IN (VALUES (?, ?), (?, ?), (?, ?))
        out = sql_to_df(
            database=database,
            sql=f"""SELECT * FROM {table} WHERE ({keys}) IN (VALUES {placeholders})""",
            params=flattened_values,
        )
    except:
        # this reads everything in memory which is less efficient but more robust
        out = sql_to_df(
            database=database,
            sql=f"SELECT * FROM {table}",
        )

    if not is_rm_duplicates:
        out = df.merge(out, on=list(df.columns), how="left")

    return out


def save_recovery(df_delta, table_name, index=False, recovery_path="."):
    """Save the rows that failed to be added to the database"""

    # saves the error rows to a csv file to avoid losing the data
    random_idx = random.randint(10**5, 10**6)
    recovery_all_path = Path(recovery_path) / f"failed_add_to_{table_name}_all_{random_idx}.csv"

    # save json as a list of dict if you don't want to keep index, else dict of dict
    orient = "index" if index else "records"
    df_delta.to_json(recovery_all_path, orient=orient, indent=2)
    logging.error(
        f"Failed to add {len(df_delta)} rows to {table_name}."
        f"Dumping all the df that you couldn't save to {recovery_all_path}"
    )


def append_df_to_db(
    df_to_add,
    database,
    table_name,
    index=False,
    recovery_path=".",
    is_prepare_to_add_to_db=True,
    is_avoid_infinity=True,
):
    """Add a dataframe to a table in a SQLite database, with recovery in case of failure.

    Parameters
    ----------
    df_to_add : pd.DataFrame
        Dataframe to add to the database.

    database : str
        Path to the database.

    table_name : str
        Name of the table to add the dataframe to.

    index : bool, optional
        Whether to add the index of the dataframe as a column.

    recovery_path : str, optional
        Path to the folder where to save the error rows in case of failure.

    is_prepare_to_add_to_db : bool, optional
        Whether to clean the dataframe before adding it to the database. Specifically will drop duplicates and
        remove columns that are not in the database.

    is_avoid_infinity : bool, optional
        Whether to replace infinity values with NaNs before adding the dataframe to the database. THis is useful because
        sqlite seems to have issues with infinity values.
    """
    if is_avoid_infinity:
        df_to_add = df_to_add.replace([np.inf, -np.inf], np.nan)

    if is_prepare_to_add_to_db:
        try:
            # this removes exact duplicates and columns not in the database
            df_delta = prepare_to_add_to_db_sql(
                df_to_add=df_to_add,
                database=database,
                sql_already_annotated=f"SELECT * FROM {table_name}",
                is_check_unique_primary_key=True,  # raise if primary key is not unique
                table_name=table_name,
            )
        except Exception as e:
            save_recovery(df_to_add, table_name, index=index, recovery_path=recovery_path)
            raise e
    else:
        df_delta = df_to_add

    with create_connection(database) as conn:
        try:
            df_delta.to_sql(table_name, conn, if_exists="append", index=index)
            logging.info(f"Added {len(df_delta)} rows to {table_name}")

        except Exception as e:
            save_recovery(df_delta, table_name, index=index, recovery_path=recovery_path)
            raise e


@contextmanager
def create_connection(database, timeout=5.0, is_print=True, **kwargs):
    """Create a database connection to a SQLite database"""
    conn = None
    try:
        conn = sqlite3.connect(database, timeout=timeout, **kwargs)
        if is_print:
            logging.info(f"Connected to {database} SQLite")
        yield conn
    except sqlite3.Error:
        logging.exception("Failed to connect with sqlite3 database:")
    finally:
        if conn:
            conn.close()


def execute_sql(database, sql):
    """Execute a sql command on a database"""
    with create_connection(database) as conn:
        c = conn.cursor()
        c.executescript(sql)
        conn.commit()


def get_all_tables(database):
    """Get all the tables in a database"""
    with create_connection(database) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        cursor.close()
    return [t[0] for t in tables]


def get_all_views(database):
    """Get all the tables in a database"""
    with create_connection(database) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
        tables = cursor.fetchall()
        cursor.close()
    return [t[0] for t in tables]
