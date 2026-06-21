from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool


SQLITE_TIMEOUT_SECONDS = 30
SQLITE_LOCK_RETRY_ATTEMPTS = 3
SQLITE_LOCK_RETRY_DELAY_SECONDS = 0.2


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "memory_agent.db"

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={
        "check_same_thread": False,
        "timeout": SQLITE_TIMEOUT_SECONDS,
    },
    poolclass=StaticPool,
    pool_pre_ping=True,
    future=True,
)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)
Base = declarative_base()


@event.listens_for(engine, "connect")
def set_sqlite_pragmas(dbapi_connection, connection_record) -> None:  # pragma: no cover - db hook
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.execute(f"PRAGMA busy_timeout={SQLITE_TIMEOUT_SECONDS * 1000};")
    cursor.close()
