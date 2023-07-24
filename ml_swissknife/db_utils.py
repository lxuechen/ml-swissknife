"""
All helper functions for database. This should work with sqlite / postgres / MySQL / etc.
Note that if you want in memory database you can just use ":memory:" as the database name.
"""
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
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import pandas as pd
import psycopg2
import sqlalchemy as sa
import tqdm
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError

from ml_swissknife.types import PathOrIOBase

logging.basicConfig(level=logging.INFO)

ENGINE_REGISTRY = {}
INITIAL_PROCESS_ID = os.getpid()


def get_engine(
    database: Union[str, sa.engine.base.Engine],
    is_use_cached_engine=True,
    is_return_is_created_new_engine=False,
    **engine_kwargs,
) -> Union[sa.engine.base.Engine, Tuple[sa.engine.base.Engine, bool]]:
    """Return the database engine to a database.

    Parameters
    ----------
    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine.

    is_use_cached_engine : bool, optional
        Whether to use a cached engine if it exists.

    is_return_is_created_new_engine : bool, optional
        Whether to return a boolean indicating whether a new engine was created.

    engine_kwargs :
        Additional arguments to `create_engine`.
    """
    is_created_new_engine = False
    if isinstance(database, sa.engine.base.Engine):
        # if engine is already created, return it
        engine = database
    else:
        if is_use_cached_engine and (os.getpid() != INITIAL_PROCESS_ID):
            logging.warning(
                "The current process is different from initial caching process. Maybe because you use multiprocessing"
                " which is not recommended with engine caching. Setting is_use_cached_engine=False to be safe."
            )
            is_use_cached_engine = False
            # in this case you likely also want to use poolclass=NullPool

        if database in ENGINE_REGISTRY and is_use_cached_engine:
            engine = ENGINE_REGISTRY[database]

        else:
            try:
                engine = sa.create_engine(database, **engine_kwargs)
            except sa.exc.ArgumentError as e:
                db_as_path = Path(database).expanduser()
                if db_as_path.is_file():
                    # converts path to sqlite url
                    engine = sa.create_engine(f"sqlite:///{db_as_path.resolve()}", **engine_kwargs)
                else:
                    raise e

            logging.info(f"Created engine for {database} which is a {engine.dialect.name} DB.")
            is_created_new_engine = True

            if is_use_cached_engine:
                ENGINE_REGISTRY[database] = engine

    if is_return_is_created_new_engine:
        return engine, is_created_new_engine
    else:
        return engine


@contextmanager
def create_engine(
    database: Union[str, sa.engine.base.Engine],
    is_use_cached_engine=True,
    **engine_kwargs,
) -> sa.engine.base.Engine:
    """Return the engine to a database.

    Parameters
    ----------
    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine, or the path if it's sqlite. If you want to use
        in memory database, you can use ":memory:".

    is_use_cached_engine : bool, optional
        If True, will use the cached engine if it exists. If False, will create a new engine.

    **engine_kwargs :
        Additional arguments to pass to `sqlalchemy.create_engine`.
    """
    is_created_new_engine = False
    try:
        engine, is_created_new_engine = get_engine(
            database,
            is_use_cached_engine=is_use_cached_engine,
            is_return_is_created_new_engine=True,
            **engine_kwargs,
        )
        yield engine
    finally:
        if is_created_new_engine:
            # only dispose if we created the engine
            engine.dispose()


@contextmanager
def create_connection(
    database: Union[str, sa.engine.base.Engine],
    is_use_cached_engine=True,
    **engine_kwargs,
) -> sa.engine.base.Connection:
    """Return the connection to a database.

    Parameters
    ----------
    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine, or the path if it's sqlite. If you want to use
        in memory database, you can use ":memory:".

    is_use_cached_engine : bool, optional
        If True, will use the cached engine if it exists. If False, will create a new engine.

    **engine_kwargs :
        Additional arguments to pass to `sqlalchemy.create_engine`.
    """
    with create_engine(database, is_use_cached_engine=is_use_cached_engine, **engine_kwargs) as engine:
        connection = None
        try:
            connection = engine.connect()
            yield connection
        finally:
            if connection is not None:
                connection.close()


def sql_to_df(
    database: Union[str, sa.engine.base.Engine],
    sql: str,
    **read_sql_kwargs,
) -> pd.DataFrame:
    """Read a SQL query into a Pandas dataframe.

    Parameters
    ----------
    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine.

    sql : str
        The SQL query to execute.

    Examples
    --------
    >>> df = sql_to_df(sql="SELECT * FROM likert_annotations LIMIT 3",  database="instruction_following.sqlite")
    >>> print(df.to_string(max_colwidth=5))
       input_id  output_id annotator  likert_score  likert_score_1_logprob  likert_score_2_logprob  likert_score_3_logprob  likert_score_4_logprob auto_raw_inputs auto_raw_outputs  auto_total_tokens
    0     0         1       chen         1           NaN                     NaN                     NaN                     NaN                    None            None              NaN
    1     0         1       t...         1           NaN                     NaN                     NaN                     NaN                    None            None              NaN
    2     0         1       t...         1           NaN                     NaN                     NaN                     NaN                    None            None              NaN
    """
    with create_engine(database) as engine:
        # could make faster using fetchmany
        df = pd.read_sql(sql=sql, con=engine, **read_sql_kwargs)

        # in the following we enforce the types of the columns in the dataframe to be the same as the types in the
        # database this is useful for example if a column is float but the rows you read are all None. In this case, the
        # dataframe will by default be of type Object and not float (because it sees all None). In reality those None
        # should be NaN and the column should be float. If you don't do that this can cause problems later on

        # parse sql to get the table name from which we read the query. THis is dirty but I didn't find a way to do
        # it using sqlalchemy (surprisingly)
        all_table_and_views = get_all_table_names(engine) + get_all_view_names(engine)
        all_tables_refered_in_sql = [t for t in sql.split() if t in all_table_and_views]

        if len(all_tables_refered_in_sql) == 1:
            table_name_to_enforce_types = all_tables_refered_in_sql[0]

        else:
            table_name_to_enforce_types = None
            logging.warning(
                f"We cannot enforce column types in df given that there are multiple tables in the "
                f"query. You should be fine, but this can be an issue if a float column has only NaN"
            )

        if table_name_to_enforce_types is not None:
            db_table = sa.Table(table_name_to_enforce_types, sa.MetaData(), autoload_with=engine)
            _enforce_type_cols_df_(df, db_table)

    return df


def delete_rows_from_db(
    database: Union[str, sa.engine.base.Engine],
    table_name: str,
    df: pd.DataFrame,
    chunksize=10000,
):
    """Delete rows from a table in a SQLite database based on the values of a dataframe.

    Parameters
    ----------
    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine.

    table_name : str
        The name of the table to delete rows from.

    df : pd.DataFrame
        The dataframe containing the rows to delete.

    Examples
    --------
    >>> df = sql_to_df(sql="SELECT * FROM likert_annotations LIMIT 3", database=database)
    >>> len(df)
    3
    >>> delete_rows_from_db(database=database,  df=df[["input_id", "output", "annotator"]], table_name="likert_annotations")
    >>> len(get_values_from_keys(database=database,  df=df[["input_id", "output", "annotator"]], table_name="likert_annotations"))
    0
    """
    with create_engine(database) as engine:
        db_table = sa.Table(table_name, sa.MetaData(), autoload_with=engine)

        for df_chunk in _dataframe_chunk_generator(df, chunksize=chunksize):
            sql_where = get_sql_where_from_df(engine, table=db_table, df=df_chunk)
            delete = db_table.delete().where(sql_where)

            execute_sql(database=engine, sql=delete)


def get_values_from_keys(
    database: Union[str, sa.engine.base.Engine],
    table_name: str,
    df: pd.DataFrame,
    chunksize: int = 10000,
    max_rows_in_memory: int = int(1e9),
    min_rows_in_memory: int = 1000,
    is_logging: bool = True,
    is_drop_JSONB: bool = True,
) -> pd.DataFrame:
    """Given a dataframe containing the primary keys of a table_name, will return the corresponding rows.

    Parameters
    ----------
    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine.

    table_name : str
        The name of the table to get the rows from.

    df : pd.DataFrame
        The dataframe containing the keys to select from. Every column treated as a key. We remove duplicates.

    chunksize : int, optional
        The number of rows to select at a time. This is useful if you have a large number of rows to select.
        E.g. it seems sql aclehmy doesn't work well with huge queries at once.

    max_rows_in_memory : int, optional
        Maximum size of table in DB that you allow yourself to load in memory. If the table is larger than this, it will
        have to use chunking and querying on the server.

    min_rows_in_memory : int, optional
        Minimum number of rows in dataframe before which you should prefer loading all table in memory rather than
        using SQL.

    is_drop_JSONB : bool, optional
        If True, will drop JSONB columns from the dataframe.

    Examples
    --------
    >>> df = sql_to_df(sql="SELECT * FROM likert_annotations LIMIT 3", database=database)
    >>> len(df)
    3
    >>> delete_rows_from_db(database=database,  df=df[["input_id", "output_id", "annotator"]], table_name="likert_annotations")
    >>> len(get_values_from_keys(database=database,  df=df[["input_id", "output_id", "annotator"]], table_name="likert_annotations"))
    0
    """
    df = df.drop_duplicates()

    with create_engine(database) as engine:
        db_table = sa.Table(table_name, sa.MetaData(), autoload_with=engine)
        n_db_rows = count_rows(engine, table_name)

        if n_db_rows == 0:
            # avoid creating large query if empty
            out = pd.DataFrame(columns=db_table.columns.keys())

        elif len(df) > min_rows_in_memory and n_db_rows < max_rows_in_memory:
            if is_logging:
                logging.info(f"Loading {n_db_rows} rows of {table_name} in memory.")

            # if the query is large and the table is manageable, then it can be much faster in memory
            df_db = sql_to_df(engine, f"SELECT * FROM {table_name}")
            out = df.merge(df_db, on=list(df.columns), how="inner")

        else:
            if is_logging:
                logging.info(f"Querying {table_name} in chunks of {chunksize} rows.")

            outs = []
            for df_chunk in _dataframe_chunk_generator(df, chunksize=chunksize):
                sql_where = get_sql_where_from_df(engine, table=db_table, df=df_chunk)
                select = sa.select(db_table).where(sql_where)

                with create_connection(engine) as connection:
                    curr_out = pd.read_sql(sql=select, con=connection)

                outs.append(curr_out)

            if len(outs) == 0:
                out = pd.DataFrame(columns=db_table.columns.keys())
            else:
                out = pd.concat(outs, ignore_index=True)

    if is_drop_JSONB:
        out = out.drop(columns=[col for col in out.columns if isinstance(db_table.columns[col].type, JSONB)])

    _enforce_type_cols_df_(out, db_table)

    return out


def append_df_to_db(
    df_to_add: pd.DataFrame,
    database: Union[str, sa.engine.base.Engine],
    table_name: str,
    index: bool = False,
    recovery_path: PathOrIOBase = ".",
    is_prepare_to_add_to_db: bool = True,
    commit_every_n_rows: Optional[int] = None,
    is_skip_writing_errors: bool = False,
    is_keep_all_columns_from_db: bool = True,
    n_retries_unique: int = 5,
    **to_sql_kwargs,
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

    commit_every_n_rows : int, optional
        The number of rows that you should have a try except over. This is useful so that if you have an error in one
        row you still add rows in other chunks. If None tries to add everything at once.

    is_skip_writing_errors : bool, optional
        Whether to skip any encountered errors when writing to db. This is useful when you are using `commit_every_n_rows`
        and still want to continue writing.

    n_retries_unique : int, optional
        Number of retries if you have errors.

    **to_sql_kwargs :
        Additional arguments to `to_sql_kwargs`.
    """
    rows_added = 0

    for df_chunk in _dataframe_chunk_generator(df_to_add, chunksize=commit_every_n_rows):
        if is_prepare_to_add_to_db:
            # this removes exact duplicates and columns not in the database
            df_delta, df_to_add_primary_key_duplicates = prepare_to_add_to_db(
                df_to_add=df_chunk,
                database=database,
                is_keep_all_columns_from_db=is_keep_all_columns_from_db,
                is_return_non_unique_primary_key=True,
                table_name=table_name,
            )
            if len(df_to_add_primary_key_duplicates) > 0:
                # save the rows that have duplicated primary keys but not other columns
                _save_recovery(
                    df_to_add_primary_key_duplicates,
                    table_name,
                    index=index,
                    recovery_path=recovery_path,
                )
        else:
            df_delta = df_chunk

        try:
            with create_connection(database) as conn:
                if conn.engine.dialect.name == "postgres":
                    to_sql_kwargs["chunksize"] = 10000
                    to_sql_kwargs["method"] = "multi"
                df_delta.to_sql(table_name, conn, if_exists="append", index=index, **to_sql_kwargs)
                rows_added += len(df_delta)

        except Exception as e:
            # the only errors you want to retry on are the one where primary key is not unique because
            # prepare_to_add_to_db should filter those
            is_unique_error = isinstance(e, IntegrityError) and isinstance(e.orig, psycopg2.errors.UniqueViolation)
            if is_unique_error and n_retries_unique > 0:
                logging.info(f"You encountered error: {e}. \n\nRetrying appending {n_retries_unique} more times")
                append_df_to_db(
                    df_to_add=df_delta,
                    database=database,
                    table_name=table_name,
                    index=index,
                    recovery_path=recovery_path,
                    is_prepare_to_add_to_db=is_prepare_to_add_to_db,
                    commit_every_n_rows=commit_every_n_rows,
                    is_skip_writing_errors=is_skip_writing_errors,
                    is_keep_all_columns_from_db=is_keep_all_columns_from_db,
                    n_retries_unique=n_retries_unique - 1,
                    **to_sql_kwargs,
                )
                rows_added += len(df_delta)
            else:
                _save_recovery(
                    df_delta,
                    table_name,
                    index=index,
                    recovery_path=recovery_path,
                    error=e,
                )
                if not is_skip_writing_errors:
                    raise e

    logging.info(f"Added {rows_added} rows to {table_name}")


def get_primary_keys(database: Union[str, sa.engine.base.Engine], table_name: str) -> list[str]:
    """Get the primary keys of a table in a database.

    Parameters
    ----------
    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine.

    table_name : str
        The name of the table to get the primary keys from.
    """
    with create_engine(database) as engine:
        inspector = sa.inspect(engine)
        return inspector.get_pk_constraint(table_name)["constrained_columns"]


### Secondary helpers ###
def get_table_info(database: Union[str, sa.engine.base.Engine], table_name: str) -> pd.DataFrame:
    """Return a dataframe with the table information of table_name in a database."""
    with create_engine(database) as engine:
        table = sa.Table(table_name, sa.MetaData(), autoload_with=engine)

    data = {
        "name": [],
        "type": [],
        "primary_key": [],
        "nullable": [],
        "default": [],
        "autoincrement": [],
        "unique": [],
    }

    for column in table.columns:
        data["name"].append(column.name)
        data["type"].append(column.type)
        data["primary_key"].append(column.primary_key)
        data["nullable"].append(column.nullable)
        data["default"].append(column.default)
        data["autoincrement"].append(column.autoincrement)
        data["unique"].append(column.unique)

    return pd.DataFrame(data)


def prepare_to_add_to_db(
    df_to_add: pd.DataFrame,
    database: Union[str, sa.engine.base.Engine],
    table_name: str,
    is_keep_all_columns_from_db: bool = True,
    is_return_non_unique_primary_key: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Prepare a dataframe to be added to a table in a database by removing rows already in the database.

    Parameters
    ----------
    df_to_add : pd.DataFrame
        The dataframe to add to the database.

    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine.

    table_name : str, optional
        The name of the table in the database to check for existing rows.

    is_keep_all_columns_from_db : bool, optional
        Whether to return all columns in DB (or only the columns in the dataframe to add).

    is_return_non_unique_primary_key : bool, optional
        Whether to return the rows you tried with primary keys that already exist in the database but have
        different values for non-primary keys. This will return a tuple of dataframes, the first being the
        dataframe to add, and the second being the rows that already exist in the database.
    """
    with create_engine(database) as engine:
        primary_keys = get_primary_keys(engine, table_name)
        try:
            df_keys = df_to_add[primary_keys]
        except KeyError:
            logging.warning(
                f"Could not find primary key in dataframe to add => assumes you want to filter on"
                f" all columns {df_to_add.columns}. Given that it's not a primary key, we deactivate"
                " checking of non-unique secondary key for unique primary by setting is_keep_all_columns_from_db=False"
            )
            df_keys = df_to_add
            primary_keys = df_to_add.columns
            # by removing other keys we ensure that check is only on the given keys
            is_keep_all_columns_from_db = False

        df_db = get_values_from_keys(database=engine, table_name=table_name, df=df_keys)

    columns = [c for c in df_db.columns if c in df_to_add.columns]
    df_to_add = df_to_add[columns]
    if not is_keep_all_columns_from_db:
        df_db = df_db[columns]

    if df_db.empty:
        if is_return_non_unique_primary_key:
            return df_to_add, df_to_add.iloc[0:0]
        return df_to_add

    df_all = pd.concat([df_db, df_to_add]).drop_duplicates()

    # Check for duplicates based on primary keys
    is_primary_key_duplicates = df_all.duplicated(subset=primary_keys, keep="first")
    if is_primary_key_duplicates.any():
        n_duplicates = is_primary_key_duplicates.sum()

        # for logging also shows the rows in the db
        is_primary_key_duplicates_all = df_all.duplicated(subset=primary_keys, keep=False)
        grouped = df_all[is_primary_key_duplicates_all].groupby(primary_keys)
        example_primary_key_duplicates = df_all.groupby(primary_keys).get_group(list(grouped.groups.keys())[0])
        logging.warning(
            f"Trying to add {n_duplicates} rows with primary keys {primary_keys} that already exist in the "
            f"database but have different values for non-primary keys. Example:\n {example_primary_key_duplicates}"
        )

    df_try_added_primary_key_duplicates = df_all[is_primary_key_duplicates]
    df_all = df_all[~is_primary_key_duplicates]  # remove the rows whose primary keys already exist in the database

    # Remove rows that are already in the database
    df_delta = get_delta_df(df_all, df_db)

    if is_return_non_unique_primary_key:
        return df_delta, df_try_added_primary_key_duplicates

    return df_delta


def get_delta_df(df_all: pd.DataFrame, df_subset: pd.DataFrame) -> pd.DataFrame:
    """return the complement of df_subset"""
    columns = list(df_all.columns)
    df_ind = df_all.merge(df_subset.drop_duplicates(), on=columns, how="left", indicator=True)
    return df_ind.query("_merge == 'left_only' ")[columns]


def get_sql_where_from_df(
    database: Union[str, sa.engine.base.Engine],
    table: Union[str, sa.Table],
    df: pd.DataFrame,
    is_str: bool = False,
) -> Union[str, sa.sql.selectable.Select]:
    """Given a dataframe of rows to select on from table_name, will return the corresponding select statement.

    Parameters
    ----------
    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine.

    table : str or Table
        The name of the table in the database to check for existing rows, or the sqlalchemy Table.

    df : pd.DataFrame
        The dataframe of rows to select on from table_name.

    is_str : bool, optional
        Whether to return the string of the SQL statement or the sqlalchemy select statement.
    """
    with create_engine(database) as engine:
        # Reflect the table
        if isinstance(table, str):
            db_table = sa.Table(table_name, sa.MetaData(), autoload_with=engine)
        else:
            db_table = table

        # Create a SELECT statement using and_ and or_
        conditions = [sa.and_(*[db_table.c[key] == value for key, value in row.items()]) for _, row in df.iterrows()]
        where_clause = sa.or_(*conditions)

        if is_str:
            return str(where_clause.compile(engine, compile_kwargs={"literal_binds": True}))
        return where_clause


def execute_sql(
    database: Union[str, sa.engine.base.Engine],
    sql: Union[str, sa.sql.expression.Executable],
    parameters=None,
    execution_options=None,
):
    """Execute a sql command on a database"""
    if isinstance(sql, str):
        sql = sa.text(sql)

    with create_connection(database) as conn:
        conn.execute(sql, parameters=parameters, execution_options=execution_options)
        conn.commit()


def get_all_table_names(database: Union[str, sa.engine.base.Engine]) -> list[str]:
    """Get all the tables in a database"""
    with create_engine(database) as engine:
        inspector = sa.inspect(engine)
        return inspector.get_table_names()


def get_all_view_names(database: Union[str, sa.engine.base.Engine]) -> list[str]:
    """Get all the views in a database"""
    with create_engine(database) as engine:
        inspector = sa.inspect(engine)
        return inspector.get_view_names()


def count_rows(database: Union[str, sa.engine.base.Engine], table_name: str) -> int:
    """Count the number of rows in a table"""
    with create_connection(database) as conn:
        row_count = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    return row_count


### private, not meant to be used
def _enforce_type_cols_df_(df: pd.DataFrame, db_table: sa.Table):
    """Inplace enforce the types of the columns in the dataframe to be the same as the types in the database."""
    for col in df.columns:
        if col in db_table.columns:
            try:
                sql_type = db_table.columns[col].type

                if isinstance(sql_type, sa.types.NullType):
                    # if column has to type we can't enforce it. This happens when some columns were added manually
                    continue

                elif sql_type.python_type in [int, float]:
                    # issue with
                    df[col] = df[col].apply(pd.to_numeric, errors="coerce")

                # don't try conversion of bool and str because will convert None to "None" and False respectively
                # else:
                #     df[col] = df[col].astype(db_table.columns[col].type.python_type)

            except Exception as e:
                # this should not happen, but worse case scenario we just don't enforce the type
                logging.warning(
                    f"Could not enforce type of column {col} in dataframe to be the same as in the database. Error: {e}"
                )


def _save_recovery(
    df_delta: pd.DataFrame,
    table_name: str,
    index: bool = False,
    recovery_path: PathOrIOBase = ".",
    error=None,
):
    """Save the rows that failed to be added to the database"""

    # saves the error rows to a csv file to avoid losing the data
    rng = random.Random(time.time())
    random_idx = rng.randint(10**5, 10**6)
    recovery_all_path = Path(recovery_path) / f"failed_add_to_{table_name}_all_{random_idx}.json"

    # save json as a list of dict if you don't want to keep index, else dict of dict
    orient = "index" if index else "records"
    df_delta.to_json(recovery_all_path, orient=orient, indent=2)
    logging.warning(
        f"Failed to add {len(df_delta)} rows to {table_name}."
        f"Dumping all the df that you couldn't save to {recovery_all_path}"
    )
    if error is not None:
        logging.warning(f"Error we are skipping: {error}")


def _dataframe_chunk_generator(df: pd.DataFrame, chunksize: Optional[int] = None, is_force_tqdm: bool = False):
    if chunksize is None:
        chunksize = max(1, len(df))

    if chunksize >= len(df) and not is_force_tqdm:
        iterator = range(0, len(df), chunksize)
    else:
        iterator = tqdm.tqdm(range(0, len(df), chunksize))

    n_iter = len(df) // chunksize

    for i in iterator:
        df_chunk = df.iloc[i : i + chunksize]

        # if many iterations then better to copy the dataframe to avoid memory issues
        if n_iter > 1:
            df_chunk = df_chunk.copy()

        yield df_chunk


def sequence_to_in_sql(sequence: Sequence):
    """Converts a sequence (eg list) to a string that can be used in an SQL query. EG
    "WHERE dataset IN {sequence_to_in_sql(['train', 'test'])}"
    """

    # code will fail if sequence has a string containing ,) => assert that
    assert len([e for e in sequence if isinstance(e, str) and ",)" in e]) == 0

    # the following line deals with singleton and with the case where you have tuples of tuples => do not change unless
    # you understand all subcases (eg if stamtement on length doesn't work)
    return str(tuple(sequence)).replace(",)", ")")  # (item,) -> (item)
