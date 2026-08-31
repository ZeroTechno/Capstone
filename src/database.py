import json
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./data/app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ImageModel(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    file_path = Column(String)
    subject = Column(String, index=True, nullable=True)
    category = Column(String, index=True, nullable=True)
    attributes_json = Column(Text, default="[]")
    caption = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    is_flagged = Column(Boolean, default=False)
    embedding_json = Column(Text, nullable=True)  # JSON-serialized list of floats
    status = Column(String, default="pending")  # pending, processed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    @property
    def attributes(self):
        return json.loads(self.attributes_json) if self.attributes_json else []

    @property
    def embedding(self):
        return json.loads(self.embedding_json) if self.embedding_json else []


class PostModel(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)
    target_subject = Column(String, nullable=True)
    embedding_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class CostLogModel(Base):
    __tablename__ = "cost_logs"

    id = Column(Integer, primary_key=True, index=True)
    operation = Column(String)  # 'vision' or 'embedding'
    model = Column(String)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    estimated_cost_usd = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(bind=engine)
