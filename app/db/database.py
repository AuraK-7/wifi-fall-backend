from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


connect_args = (
    {"check_same_thread": False}
    if settings.DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_sqlite_schema_compatibility() -> None:
    """Add columns that older local SQLite databases may be missing."""
    if not settings.DATABASE_URL.startswith("sqlite"):
        return

    inspector = inspect(engine)
    if "alert_events" not in inspector.get_table_names():
        return

    existing_columns = {
        column["name"]
        for column in inspector.get_columns("alert_events")
    }
    required_columns = {
        "analytics_snapshot": "JSON",
        "frame_id": "INTEGER",
        "evidence_chain": "JSON",
        "source": "VARCHAR(50)",
    }

    with engine.begin() as connection:
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    text(f"ALTER TABLE alert_events ADD COLUMN {column_name} {column_type}")
                )
                if column_name == "source":
                    connection.execute(
                        text("UPDATE alert_events SET source = 'unknown' WHERE source IS NULL")
                    )
