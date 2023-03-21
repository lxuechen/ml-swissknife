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
) -> pd.DataFrame:
    """Prepare a dataframe to be added to a table in a SQLite database. by removing rows already in the database.

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

    """
    df_db = sql_to_df(
        database=database,
        sql=sql_already_annotated,
    )
    columns = [c for c in df_db.columns if c in df_to_add.columns]
    if not is_keep_all_columns_from_db:
        df_db = df_db[columns]

    df_all = pd.concat([df_db, df_to_add[columns]]).drop_duplicates()
    df_delta = get_delta_df(df_all, df_db)
    return df_delta


# depreciate once merged
def prepare_to_add_to_db(df_to_add, database, table_name, is_subset_columns=False) -> pd.DataFrame:
    """Prepare a dataframe to be added to a table in a SQLite database. by removing rows already in the database.
    and columns not in the database."""
    with create_connection(database) as conn:
        df_db = pd.read_sql(f"SELECT * FROM {table_name}", conn)
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
            sql=f"SELECT * FROM {table} WHERE ({keys}) IN (VALUES {placeholders})",
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


def append_df_to_db(
    df_to_add,
    database,
    table_name,
    index=False,
    recovery_path=".",
    is_prepare_to_add_to_db=True,
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
    """
    if is_prepare_to_add_to_db:
        df_delta = prepare_to_add_to_db_sql(
            df_to_add=df_to_add,
            database=database,
            sql_already_annotated=f"SELECT * FROM {table_name}",
        )
    else:
        df_delta = df_to_add

    with create_connection(database) as conn:
        try:
            df_delta.to_sql(table_name, conn, if_exists="append", index=index)
            logging.info(f"Added {len(df_delta)} rows to {table_name}")
        except sqlite3.Error as e:
            # if there is an error, it tries to add the rows one by one
            rows_errors = []
            for i in range(len(df_delta)):
                try:
                    df_delta.iloc[i : i + 1].to_sql(table_name, conn, if_exists="append", index=index)
                except:
                    rows_errors.append(i)
            logging.error(
                f"Failed to add {len(rows_errors)} rows out of {len(df_delta)} to {table_name} with error: {e}"
            )

            # saves the error rows to a csv file to avoid losing the data
            df_errors = df_delta.iloc[rows_errors]
            random_idx = random.randint(10**5, 10**6)
            recovery_error_path = Path(recovery_path) / f"failed_add_to_{table_name}_errors_{random_idx}.csv"
            recovery_all_path = Path(recovery_path) / f"failed_add_to_{table_name}_all_{random_idx}.csv"

            # save json as a list of dict if you don't want to keep index, else dict of dict
            orient = "index" if index else "records"
            df_errors.to_json(recovery_error_path, orient=orient, indent=2)
            df_to_add.to_json(recovery_all_path, orient=orient, indent=2)
            logging.error(f"Saved errors to {recovery_error_path} and all df to {recovery_all_path}")


@contextmanager
def create_connection(db_file, timeout=5.0, is_print=True, **kwargs):
    """Create a database connection to a SQLite database"""
    conn = None
    try:
        conn = sqlite3.connect(db_file, timeout=timeout, **kwargs)
        if is_print:
            logging.info(f"Connected to {db_file} SQLite")
        yield conn
    except sqlite3.Error:
        logging.exception("Failed to connect with sqlite3 database:")
    finally:
        if conn:
            conn.close()


def create_table(conn, create_table_sql):
    """Create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error:
        logging.exception("Failed to connect with sqlite3 database:")
