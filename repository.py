from datetime import datetime, timedelta
import json
from sqlalchemy.orm import Session
from models import PredictionSession,Users,DetectionObjects
from sqlalchemy import and_, delete, distinct, func, join, select

def save_prediction_session(uid, original_image, predicted_image,username,db: Session):
    """
    Save prediction session to database
    """
    row = PredictionSession(uid=uid, predicted_image=predicted_image, original_image=original_image,username=username)
    # Add the instance to the session and commit
    db.add(row)
    db.commit()

def save_detection_object(prediction_uid, label, score, box,db:Session):
    """
    Save detection object to database
    """
    row=DetectionObjects(prediction_uid=prediction_uid,label=label,score=score,box=json.dumps(box))
    db.add(row)
    db.commit()

def query_user_by_credentials(db: Session, username, password):
    return db.query(Users).filter_by(username=username, password=password).first()

def query_prediction_count(db: Session,username):
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    count = db.query(func.count(PredictionSession.uid)).filter(and_(PredictionSession.timestamp >= seven_days_ago,PredictionSession.username == username)).scalar()
    return count

def query_unique_labels(db: Session,username):
    seven_days_ago = datetime.utcnow() - timedelta(days=7)

    stmt = (
        select(distinct(DetectionObjects.label))
        .select_from(
            join(DetectionObjects, PredictionSession, DetectionObjects.prediction_uid == PredictionSession.uid)
        )
        .where(and_(PredictionSession.timestamp >= seven_days_ago ,PredictionSession.username==username))
    )

    result = db.execute(stmt).scalars().all()
    return result

def query_delete_from(db:Session,db_name,uid,username):
    if db_name=='PredictionSession':
        stmt=delete(PredictionSession).where(and_(PredictionSession.uid==uid,PredictionSession.username==username))
        result=db.execute(stmt)
        db.commit()
        return result.rowcount
    stmt=delete(DetectionObjects).where(DetectionObjects.prediction_uid==uid)
    result=db.execute(stmt)
    db.commit()
    return result.rowcount

def query_get_prediction_by_uid(uid, db_name, db: Session,username):
    if db_name=='PredictionSession':
        result=db.query(PredictionSession).filter_by(uid=uid,username=username).first()
        return result
    result=db.query(DetectionObjects).filter_by(prediction_uid=uid).all()
    return result

def query_get_prediction_by_label(label,db: Session,username):
    rows = (
    db.query(PredictionSession.uid, PredictionSession.timestamp)
    .join(DetectionObjects, PredictionSession.uid == DetectionObjects.prediction_uid)
    .filter(DetectionObjects.label == label,PredictionSession.username==username)
    .distinct()
    .all())
    return rows

def query_get_prediction_by_score(min_score,db,username):
    rows = (
    db.query(PredictionSession.uid, PredictionSession.timestamp)
    .join(DetectionObjects, PredictionSession.uid == DetectionObjects.prediction_uid)
    .filter(DetectionObjects.score >= min_score,PredictionSession.username==username)
    .distinct()
    .all())
    return rows

def query_get_prediction_image(uid,db,username):
    row=select(PredictionSession.predicted_image).select_from(PredictionSession).where(and_(PredictionSession.uid==uid,PredictionSession.username==username))
    result = db.execute(row).first() 
    return result

def query_add_user(username,password,db):
    existing_user = db.query(Users).filter(Users.username == username).first()
    if existing_user:
        return 'Username already exists'
    row=Users(username=username,password=password)
    db.add(row)
    db.commit()
    return row


        
    



