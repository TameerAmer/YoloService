# models.py

from sqlalchemy import Column, Index, String, DateTime, Integer, Float

from datetime import datetime,timezone
from sqlalchemy.orm import declarative_base
# All models inherit from this base class
Base = declarative_base()

class PredictionSession(Base):
    """
    Model for prediction_sessions table
    
    This replaces: CREATE TABLE prediction_sessions (...)
    """
    __tablename__ = 'prediction_sessions'
    
    uid = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc))
    original_image = Column(String)
    predicted_image = Column(String)
    username=Column(String)

class DetectionObjects(Base):
    """
    CREATE TABLE detection_objects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prediction_uid TEXT,
  label TEXT,
  score REAL,
  box TEXT,
  FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
    )"""
    __tablename__='detection_objects'

    id=Column(Integer,primary_key=True)
    prediction_uid=Column(String)
    label=Column(String)
    score=Column(Float)
    box=Column(String)

class Users(Base):
    __tablename__='users'
    
    user_id=Column(Integer,primary_key=True)
    username=Column(String,unique=True)
    password=Column(String)


Index("idx_prediction_uid", DetectionObjects.prediction_uid)
Index("idx_label", DetectionObjects.label)
Index("idx_score", DetectionObjects.score)


