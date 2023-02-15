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

import sqlite3
from contextlib import contextmanager


def append_df_to_db(df, database, table_name, index=False, recovery_path="."):
    """Add a dataframe to a table in a SQLite database, with recovery in case of failure.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to add to the database.

    database : str
        Path to the database.

    table_name : str
        Name of the table to add the dataframe to.

    index : bool, optional
        Whether to add the index of the dataframe as a column.

    recovery_path : str, optional
        Path to the folder where to save the error rows in case of failure.
    """
    if not index:
        # if the index should not be considered as a column then there might be duplicates
        # so we remove them as this would cause an error
        df = df.drop_duplicates()

    with create_connection(database) as conn:
        try:
            df.to_sql(table_name, conn, if_exists="append", index=index)
            print(f"Added {len(df)} rows to {table_name}")
        except sqlite3.Error as e:
            # if there is an error, it tries to add the rows one by one
            rows_errors = []
            for i in range(len(df)):
                try:
                    df.iloc[i : i + 1].to_sql(table_name, conn, if_exists="append", index=index)
                except:
                    rows_errors.append(i)
            print(f"Failed to add {len(rows_errors)} rows out of {len(df)} to {table_name} with error: {e}")

            # saves the error rows to a csv file to avoid losing the data
            df_errors = df.iloc[rows_errors]
            recovery_path = Path(recovery_path) / f"failed_add_to_{table_name}_{random.randint(10**5, 10**6)}.csv"
            df_errors.to_csv(recovery_path, index=index)
            print(f"Saved errors to {recovery_path}")


@contextmanager
def create_connection(db_file):
    """Create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to {db_file} SQLite")
        yield conn
    except sqlite3.Error as e:
        print("Failed to connect with sqlite3 database", e)
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
    except sqlite3.Error as e:
        print(e)
