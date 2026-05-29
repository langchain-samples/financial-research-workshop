"""Shared helpers used by Module 1 (Chinook demo) and others."""

import sqlite3
import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


def show_graph(graph, xray=False):
    """Display a LangGraph mermaid diagram with ASCII fallback."""
    from IPython.display import Image
    try:
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        print(f"Image rendering failed: {e}")
        print("\nFalling back to ASCII:\n")
        print(graph.get_graph(xray=xray).draw_ascii())
        return None


def get_engine_for_chinook_db():
    """Download the Chinook SQL script and load it into an in-memory SQLite DB."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
