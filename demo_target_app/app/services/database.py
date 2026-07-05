import os
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

DataBASE_URL  = os.getenv("DATABASE_URL", "postgresql://admin:admin@postgres:5432/ragradar_telemetry")

engine = create_engine(DataBASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush = False, bind= engine)
Base = declarative_base()

class TraceRecord(Base):
    __tablename__ = "query_traces"

    trace_id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime, default = lambda:datetime.now(timezone.utc))

    question = Column(String, nullable=False)
    answer = Column(String,nullable=False)

    retrieval_mode = Column(String, nullable=False)

    chunks = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=False)

Base.metadata.create_all(bind=engine)