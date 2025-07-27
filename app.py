from fastapi import FastAPI, UploadFile, File, HTTPException, Request,Depends
from fastapi.responses import FileResponse
from fastapi.security  import HTTPBasic, HTTPBasicCredentials
from ultralytics import YOLO
from PIL import Image
import os
import uuid
import shutil
import time
from typing import Annotated

from db import get_db,engine
from models import Base
from sqlalchemy.orm import Session
import repository

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()

security=HTTPBasic()

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"
 
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")  

Base.metadata.create_all(bind=engine)

def verify_user(credentials: Annotated[HTTPBasicCredentials, Depends(security)],db: Session = Depends(get_db)):
    username = credentials.username.strip()
    password = credentials.password.strip()

    user=repository.query_user_by_credentials(db,username,password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username


#for the "Depends" statement in predict()
async def optional_auth(request: Request):
    auth = request.headers.get("Authorization")
    if not auth:
        return None  # No credentials provided
    return await security(request)

@app.post("/predict")
def predict(file: UploadFile = File(...),credentials: Annotated[str | None, Depends(optional_auth)] = None,db: Session = Depends(get_db)):
    """
    Predict objects in an image
    """
    username = None
    if credentials:
        try:
            username = verify_user(credentials,db)
        except HTTPException:
            username = None    #Invalid credentials still allow prediction, username remains null

    start_time = time.time()
    
    ext = os.path.splitext(file.filename)[1]
    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model(original_path, device="cpu")
    annotated_frame = results[0].plot()
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    repository.save_prediction_session(uid, original_path, predicted_path,username,db)
    
    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        repository.save_detection_object(uid, label, score, bbox,db)
        detected_labels.append(label)

    processing_time = round(time.time() - start_time, 2)

    return {
        "prediction_uid": uid, 
        "detection_count": len(results[0].boxes),
        "labels": detected_labels,
        "time_took": processing_time
    }


@app.get("/prediction/count")
def get_prediction_count(username: Annotated[str, Depends(verify_user)], db: Session=Depends(get_db)):
    """
    Get total number of prediction sessions
    """
    count=repository.query_prediction_count(db,username)
    return {"count": count}

@app.get("/labels")
def get_uniqe_labels(username: Annotated[str, Depends(verify_user)],db: Session = Depends(get_db)):
    """
    Get all unique labels from detection objects
    """
    labels=repository.query_unique_labels(db,username)
    return {"labels": labels}

@app.delete("/prediction/{uid}")
def delete_prediction(uid: str, username: Annotated[str, Depends(verify_user)], db: Session = Depends(get_db)):
    # First, check if the prediction session exists and belongs to the user
    dele1 = repository.query_delete_from(db, 'PredictionSession', uid, username)
    if dele1 == 0:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Delete detection objects (this might return 0 if no detections exist, which is valid)
    repository.query_delete_from(db, 'DetectionObjects', uid, username)

    # Check for the file with any of the known image extensions
    deleted = False
    for ext in [".jpg", ".jpeg", ".png"]:
        upload_path = os.path.join(UPLOAD_DIR, uid + ext)
        predict_path = os.path.join(PREDICTED_DIR, uid + ext)

        if os.path.exists(upload_path):
            os.remove(upload_path)
            deleted = True
        if os.path.exists(predict_path):
            os.remove(predict_path)
            deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail="Prediction file not found")

    return {"message": "Successfully Deleted"}

        

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str,username: Annotated[str, Depends(verify_user)],db: Session = Depends(get_db)):
    """
    Get prediction session by uid with all detected objects
    """

    # Get prediction session
    result = repository.query_get_prediction_by_uid(uid,'PredictionSession',db,username)
    if not result:
        raise HTTPException(status_code=404, detail="Prediction not found")
        
    # Get all detection objects for this prediction
    objects = repository.query_get_prediction_by_uid(uid,'DetectionObjects',db,username)

    
    return {
        "uid": result.uid,
        "timestamp": result.timestamp,
        "original_image": result.original_image,
        "predicted_image": result.predicted_image,
        "detection_objects": [
            {
                "id": obj.id,
                "label": obj.label,
                "score": obj.score,
                "box": obj.box
            } for obj in objects
        ]
    }

@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str,username: Annotated[str, Depends(verify_user)],db: Session = Depends(get_db)):
    """
    Get prediction sessions containing objects with specified label
    """
    rows=repository.query_get_prediction_by_label(label,db,username)    
    return [{"uid": row.uid, "timestamp": row.timestamp} for row in rows]

@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float,username: Annotated[str, Depends(verify_user)],db: Session=Depends(get_db)):
    """
    Get prediction sessions containing objects with score >= min_score
    """
    rows=repository.query_get_prediction_by_score(min_score,db,username)    
    return [{"uid": row.uid, "timestamp": row.timestamp} for row in rows]

@app.get("/image/{type}/{filename}")
def get_image(type: str, filename: str,credentials: Annotated[str, Depends(verify_user)]):
    """
    Get image by type and filename
    """
    if type not in ["original", "predicted"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    path = os.path.join("uploads", type, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, request: Request,username: Annotated[str, Depends(verify_user)],db: Session = Depends(get_db)):
    """
    Get prediction image by uid
    """
    accept = request.headers.get("accept", "")
    row = repository.query_get_prediction_image(uid,db,username)
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")
    image_path = row[0]

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Predicted image file not found")

    if "image/png" in accept:
        return FileResponse(image_path, media_type="image/png")
    elif "image/jpeg" in accept or "image/jpg" in accept:
        return FileResponse(image_path, media_type="image/jpeg")
    else:
        # If the client doesn't accept image, respond with 406 Not Acceptable
        raise HTTPException(status_code=406, detail="Client does not accept an image format")
    

@app.post("/register")
def register_user(credentials: Annotated[HTTPBasicCredentials, Depends(security)],db:Session=Depends(get_db)):
    username = credentials.username.strip()
    password = credentials.password.strip()

    row=repository.query_add_user(username,password,db)
    if row=='Username already exists':
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"message": "User registered successfully"}




@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__":  #paragma: no cover
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080,reload=True)