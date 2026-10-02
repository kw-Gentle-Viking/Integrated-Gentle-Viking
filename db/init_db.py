import argparse
import psycopg2


def create_database(dsn_admin: str, db_name: str = "stock_db_v2") -> None:
    conn = psycopg2.connect(dsn_admin)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        if cur.fetchone() is None:
            cur.execute(f'CREATE DATABASE "{db_name}"')
    conn.close()


def apply_schema(dsn: str, schema_path: str = "db/schema.sql") -> None:
    with open(schema_path, "r") as f:
        ddl = f.read()
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--admin-dsn", required=True, help="admin DSN, e.g. postgresql://user:pw@localhost/postgres")
    parser.add_argument("--target-dsn", required=True, help="stock_db_v2 DSN")
    args = parser.parse_args()
    create_database(args.admin_dsn)
    apply_schema(args.target_dsn)
    print("stock_db_v2 ready")
