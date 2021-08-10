import sqlite3

def fetchCidades() -> list():
    con = sqlite3.connect('database.db')
    c = con.cursor()
    x = c.execute('SELECT * from CITIES')
    print(x.fetchall())
    return x.fetchall()

fetchCidades()