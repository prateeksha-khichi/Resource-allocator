import os
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
else:
    # Use SQLite
    engine = create_engine("sqlite:///allocator.db")

Base = declarative_base()

class Episode(Base):
    __tablename__ = "episodes"
    id = Column(Integer, primary_key=True)
    task_id = Column(String(50))
    total_reward = Column(Float)
    steps = Column(Integer)
    success = Column(Boolean)
    timestamp = Column(Float)
    avg_cpu = Column(Float)
    avg_memory = Column(Float)
    cost_saved = Column(Float)

class Step(Base):
    __tablename__ = "steps"
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer)
    step_num = Column(Integer)
    observation_json = Column(Text)
    action_json = Column(Text)
    reward = Column(Float)
    explanation_json = Column(Text)

class MetricsHistory(Base):
    __tablename__ = "metrics_history"
    id = Column(Integer, primary_key=True)
    timestamp = Column(Float)
    cpu_percent = Column(Float)
    memory_percent = Column(Float)
    containers_count = Column(Integer)

class PolicyVersion(Base):
    __tablename__ = "policy_versions"
    id = Column(Integer, primary_key=True)
    version = Column(String(50))
    path = Column(String(255))
    avg_reward = Column(Float)
    timestamp = Column(Float)

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
