"""
All helper functions for database. This should work with sqlite / postgres / MySQL / etc.
Note that if you want in memory database you can just use ":memory:" as the database name.
"""

import logging
import random
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
from ml_swissknife.types import PathOrIOBase
from typing import Optional
import numpy as np
import sqlalchemy as sa
from typing import Union, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO)

@contextmanager
def create_engine(
    database: Union[str, sa.engine.base.Engine], is_print: bool = True, **engine_kwargs
) -> sa.engine.base.Engine:
    """Return the engine to a database.

    Parameters
    ----------
    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine, or the path if it's sqlite. If you want to use
        in memory database, you can use ":memory:".

    is_print : bool, optional
        Whether to print the database.

    **engine_kwargs :
        Additional arguments to pass to `sqlalchemy.create_engine`.
    """
    is_new_engine = not isinstance(database, sa.engine.base.Engine)
    if not is_new_engine:
        yield database
    else:
        try:
            engine = get_engine(database, **engine_kwargs)

            if is_print:
                logging.info(f"Connected to {database} which is a {engine.dialect.name} DB.")

            yield engine
        finally:
            # only dispose if we created the engine
            engine.dispose()


@contextmanager
def create_connection(
    database: Union[str, sa.engine.base.Engine], is_print: bool = True, **engine_kwargs
) -> sa.engine.base.Connection:
    """Return the connection to a database.

    Parameters
    ----------
    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine, or the path if it's sqlite. If you want to use
        in memory database, you can use ":memory:".

    is_print : bool, optional
        Whether to print the database.

    **engine_kwargs :
        Additional arguments to pass to `sqlalchemy.create_engine`.
    """
    with create_engine(database, is_print=is_print, **engine_kwargs) as engine:
        try:
            connection = engine.connect()
            yield connection
        finally:
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
       input_id output annotator  likert_score  likert_score_1_logprob  likert_score_2_logprob  likert_score_3_logprob  likert_score_4_logprob auto_raw_inputs auto_raw_outputs  auto_total_tokens
    0     2      I...   t...         4           NaN                     NaN                     NaN                     NaN                    None            None              NaN
    1     2      I...   t...         4           NaN                     NaN                     NaN                     NaN                    None            None              NaN
    2     2      I...   chen         2           NaN                     NaN                     NaN                     NaN                    None            None              NaN
    """
    with create_engine(database) as engine:
        if engine.dialect.name == "postgresql":
            # `method="multi"` means combines multiple rows into a single INSERT statement => faster
            # `chunksize` divides the DataFrame into smaller chunks and then insert each chunk separately. This can help
            # reduce the memory usage and improve the performance when inserting a large number of rows into the
            # database.
            read_sql_kwargs["chunksize"] = 10000
            read_sql_kwargs["method"] = "multi"

        with create_connection(engine) as conn:
            df = pd.read_sql(sql=sql, con=conn, **read_sql_kwargs)

        # in the following we enforce the types of the columns in the dataframe to be the same as the types in the
        # database this is useful for example if a column is float but the rows you read are all None. In this case, the
        # dataframe will by default be of type Object and not float (because it sees all None). In reality those None
        # should be NaN and the column should be float. If you don't do that this can cause problems later on

        # parse sql to get the table name from which we read the query
        all_table_and_views = get_all_tables(engine) + get_all_views(engine)
        all_tables_refered_in_sql = [t for t in sql.split() if t in all_table_and_views]

        if len(all_tables_refered_in_sql) == 1:
            table_name_to_enforce_types = all_tables_refered_in_sql[0]

        elif len(all_tables_refered_in_sql) > 1:
            table_name_to_enforce_types = all_tables_refered_in_sql[0]
            logging.warning(
                f"There are multiple tables you referred to, we think {table_name_to_enforce_types} "
                f"is the right one for enforcing column types in dataframe."
            )

        else:
            table_name_to_enforce_types = None
            logging.warning(
                f"There are no tables you referred to in your SQL query, we cannot enforce column types in dataframe."
            )

        if table_name_to_enforce_types is not None:
            db_table = sa.Table(table_name_to_enforce_types, sa.MetaData(), autoload_with=engine)
            enforce_type_cols_df_(df, db_table)

    return df


def delete_rows_from_db(database: Union[str, sa.engine.base.Engine], table_name: str, df: pd.DataFrame):
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

        sql_where = get_sql_where_from_df(engine, table=db_table, df=df)
        delete = db_table.delete().where(sql_where)

        execute_sql(database=engine, sql=delete)


def get_values_from_keys(
    database: Union[str, sa.engine.base.Engine], table_name: str, df: pd.DataFrame
) -> pd.DataFrame:
    """Given a dataframe containing the primary keys of a table_name, will return the corresponding rows

    Parameters
    ----------
    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine.

    table_name : str
        The name of the table to get the rows from.

    df : pd.DataFrame
        The dataframe containing the keys to select from. Every column treated as a key.

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

        sql_where = get_sql_where_from_df(engine, table=db_table, df=df)
        select = sa.select(db_table).where(sql_where)

        with create_connection(engine) as connection:
            result = connection.execute(select)
            out = pd.DataFrame(result.fetchall(), columns=result.keys())

    enforce_type_cols_df_(out, db_table)

    return out


def append_df_to_db(
    df_to_add: pd.DataFrame,
    database: Union[str, sa.engine.base.Engine],
    table_name: str,
    index: bool = False,
    recovery_path: PathOrIOBase = ".",
    is_prepare_to_add_to_db: bool = True,
    **to_sql_kwargs
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

    **to_sql_kwargs :
        Additional arguments to `to_sql_kwargs`.
    """
    if is_prepare_to_add_to_db:
        # this removes exact duplicates and columns not in the database
        df_delta, df_to_add_primary_key_duplicates = prepare_to_add_to_db(
            df_to_add=df_to_add,
            database=database,
            is_return_non_unique_primary_key=True,
            table_name=table_name,
        )
        if len(df_to_add_primary_key_duplicates) > 0:
            # save the rows that have duplicated primary keys but not other columns
            save_recovery(
                df_to_add_primary_key_duplicates,
                table_name,
                index=index,
                recovery_path=recovery_path,
            )
    else:
        df_delta = df_to_add

    try:
        with create_connection(database) as conn:
            if conn.engine.dialect.name == "postgres":
                to_sql_kwargs["chunksize"] = 10000
                to_sql_kwargs["method"] = "multi"
            df_delta.to_sql(table_name, conn, if_exists="append", index=index, **to_sql_kwargs)
            logging.info(f"Added {len(df_delta)} rows to {table_name}")

    except Exception as e:
        save_recovery(df_delta, table_name, index=index, recovery_path=recovery_path)
        raise e


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


def get_engine(database: Union[str, sa.engine.base.Engine], **engine_kwargs) -> sa.engine.base.Engine:
    """Return the database engine to a database.

    Parameters
    ----------
    database : str or engine
        The URL to the database to connect to, or the sqlalchemy engine.
    """
    if isinstance(database, sa.engine.base.Engine):
        engine = database
    else:
        try:
            engine = sa.create_engine(database, **engine_kwargs)
        except sa.exc.ArgumentError as e:
            if Path(database).is_file():
                # converts path to sqlite url
                engine = sa.create_engine(f"sqlite:///{database}", **engine_kwargs)
            else:
                raise e

    return engine


def enforce_type_cols_df_(df: pd.DataFrame, db_table: sa.Table):
    """Inplace enforce the types of the columns in the dataframe to be the same as the types in the database."""
    for col in df.columns:
        if col in db_table.columns:
            try:
                sql_type = db_table.columns[col].type

                if isinstance(sql_type, sa.types.NullType):
                    # if column has to type we can't enforce it. This happens when some columns were added manually
                    continue

                elif sql_type.python_type == int:
                    df[col] = df[col].apply(pd.to_numeric, errors="coerce")

                else:
                    df[col] = df[col].astype(db_table.columns[col].type.python_type)

            except Exception as e:
                # this should not happen, but worse case scenario we just don't enforce the type
                logging.warning(
                    f"Could not enforce type of column {col} in dataframe to be the same as in the database. Error: {e}"
                )


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

    is_keep_all_columns_from_db : bool, optional
        Whether to return all columns in DB (or only the columns in the dataframe to add).

    is_return_non_unique_primary_key : bool, optional
        Whether to return the rows you tried with primary keys that already exist in the database but have
        different values for non-primary keys. This will return a tuple of dataframes, the first being the
        dataframe to add, and the second being the rows that already exist in the database.

    table_name : str, optional
        The name of the table in the database to check for existing rows.
    """
    with create_engine(database) as engine:
        db_table = sa.Table(table_name, sa.MetaData(), autoload_with=engine)
        primary_keys = get_primary_keys(engine, table_name)

        # select all rows from the database that have the same primary keys as the rows to add
        sql_where = get_sql_where_from_df(engine, table=db_table, df=df_to_add[primary_keys])
        select = db_table.select().where(sql_where)

        with create_connection(engine) as conn:
            result = conn.execute(select)
            df_db = pd.DataFrame(result.fetchall(), columns=result.keys())

    enforce_type_cols_df_(df_db, db_table)

    columns = [c for c in df_db.columns if c in df_to_add.columns]

    if not is_keep_all_columns_from_db:
        df_db = df_db[columns]

    df_all = pd.concat([df_db, df_to_add[columns]]).drop_duplicates()

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

    df_try_added_pimary_key_duplicates = df_all[is_primary_key_duplicates]
    df_all = df_all[~is_primary_key_duplicates]  # remove the rows whose primary keys already exist in the database

    # Remove rows that are already in the database
    df_delta = get_delta_df(df_all, df_db)

    if is_return_non_unique_primary_key:
        return df_delta, df_try_added_pimary_key_duplicates

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


def save_recovery(df_delta: pd.DataFrame, table_name: str, index: bool = False, recovery_path: PathOrIOBase = "."):
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


def get_all_tables(database: Union[str, sa.engine.base.Engine]) -> list[str]:
    """Get all the tables in a database"""
    with create_engine(database) as engine:
        inspector = sa.inspect(engine)
        return inspector.get_table_names()


def get_all_views(database: Union[str, sa.engine.base.Engine]) -> list[str]:
    """Get all the views in a database"""
    with create_engine(database) as engine:
        inspector = sa.inspect(engine)
        return inspector.get_view_names()
