import sqlite3
from sqlite3 import Error


def connect(db):
    """
    :param db: Server id
    :return: connection
    """
    try:
        conn = sqlite3.connect(db)
        return conn
    except Error as e:
        print(e)
    return None


def select_all_tasks(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")

    rows = cur.fetchall()

    for row in rows:
        print(row)


def main():
    database = '/home/chasebrown/Desktop/Database/SpringCyber2019'

    # create a database connection
    conn = connect(database)

    select_all_tasks(conn)
    print('test')


if __name__ == '__main__':
    main()