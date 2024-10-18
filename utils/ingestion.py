import os
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text, inspect, Engine

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
## Data Ingestion
def data_ingestion(
    filename,
    table_name="data",
    db_url="sqlite:///data/data.db",
):
    """
    1. Read & convert data into sql
    Input: csv/excel
    Output: sql

    """
    df = pd.read_csv(filename)

    df.drop_duplicates(inplace=True)
    # df.fillna(0, inplace=True)  # Example handling missing values

    df.head()

    # convert column name with space to underscore
    # to reduce complexity on sql query generation
    df.columns = df.columns.str.replace(" ", "_")

    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    sql_database = SQLDatabase.from_uri(db_url)
    inspector = inspect(engine)
    columns_info = inspector.get_columns(table_name)

    schema_details = []
    for column in columns_info:
        col_name = column["name"]
        col_type = column["type"]
        schema_details.append(f"{col_name} ({col_type})")
    schema = "\n".join(schema_details)

    logger.info(f"{filename} Schema:\n{schema}")

    return df, sql_database, engine, schema

def data_ingestion_mysql():
    
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_port = os.getenv("DB_PORT")
    
    SQLALCHEMY_DATABASE_URL = (
        f"mysql+mysqlconnector://{db_user}:{db_password}"
        f"@{db_host}:{db_port}"
        f"/{db_name}"
    )

    db = SQLDatabase.from_uri(SQLALCHEMY_DATABASE_URL)

    return db