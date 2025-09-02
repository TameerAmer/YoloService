from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request,Depends
from fastapi.params import Query
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

# S3 additions
import boto3
from botocore.exceptions import ClientError
from mimetypes import guess_type

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

load_dotenv() 
app = FastAPI()

security=HTTPBasic()

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"


AWS_REGION = os.environ.get("AWS_REGION")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")

s3_client = None
if AWS_REGION and AWS_S3_BUCKET:
    s3_client = boto3.client("s3", region_name=AWS_REGION)


def _s3_required():
    if not s3_client or not AWS_S3_BUCKET:
        raise HTTPException(status_code=500, detail="S3 not configured (missing AWS_REGION / AWS_S3_BUCKET)")

def _upload_to_s3(local_path: str, key: str):
    _s3_required()
    ctype, _ = guess_type(local_path)
    extra = {"ContentType": ctype or "application/octet-stream"}
    try:
        s3_client.upload_file(local_path, AWS_S3_BUCKET, key, ExtraArgs=extra)
    except ClientError as e:
        raise HTTPException(status_code=502, detail=f"S3 upload failed: {e.response['Error']['Message']}")

def _download_from_s3(key: str, local_path: str):
    _s3_required()
    try:
        s3_client.download_file(AWS_S3_BUCKET, key, local_path)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            raise HTTPException(status_code=404, detail="Image key not found in S3")
        raise HTTPException(status_code=502, detail=f"S3 download failed: {e.response['Error']['Message']}")

def _object_exists(key: str) -> bool:
    _s3_required()
    try:
        s3_client.head_object(Bucket=AWS_S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def _copy_s3_object(source_key: str, dest_key: str):
    _s3_required()
    try:
        s3_client.copy(
            {"Bucket": AWS_S3_BUCKET, "Key": source_key},
            AWS_S3_BUCKET,
            dest_key
        )
    except ClientError as e:
        raise HTTPException(status_code=502, detail=f"S3 copy failed: {e.response['Error']['Message']}")
# ...existing code...
 

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
def predict(
    file: UploadFile | None = File(None),
    img: str | None = Query(default=None, description="S3 image file name (e.g. beatles.jpeg)"),
    credentials: Annotated[str | None, Depends(optional_auth)] = None,
    db: Session = Depends(get_db)
):
    """
    Predict objects in an image.
    Either upload a file OR provide ?img=<filename> to fetch from S3.
    If ?img is used, image will be downloaded from S3 at <bucket>/<user|anonymous>/original/<filename>.
    After prediction both original and annotated images are stored locally and uploaded to S3:
      <bucket>/<user|anonymous>/original/<filename>
      <bucket>/<user|anonymous>/predicted/<filename>
    """
    if not file and not img:
        raise HTTPException(status_code=400, detail="Provide an uploaded file or img query parameter")
    if file and img:
        raise HTTPException(status_code=400, detail="Use either file upload or img query parameter, not both")

    username = None
    if credentials:
        try:
            username = verify_user(credentials, db)
        except HTTPException:
            username = None  # anonymous if invalid

    user_folder = username or "anonymous"

    start_time = time.time()

    if img:
        # Use S3 image
        ext = os.path.splitext(img)[1]
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            raise HTTPException(status_code=400, detail="Unsupported image extension")
        uid = str(uuid.uuid4())
        original_path = os.path.join(UPLOAD_DIR, uid + ext)
        predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

        original_key = f"{user_folder}/original/{img}"
        predicted_key = f"{user_folder}/predicted/{img}"

        try:
            _download_from_s3(original_key, original_path)
        except HTTPException as e:
            # If authenticated user and original not found, try fallback locations
            if e.status_code == 404 and username:
                fallback_keys = [
                    f"anonymous/original/{img}",  # from anonymous pool
                    img  # root level object (legacy placement)
                ]
                copied = False
                for fk in fallback_keys:
                    if _object_exists(fk):
                        _copy_s3_object(fk, original_key)
                        _download_from_s3(original_key, original_path)
                        copied = True
                        break
                if not copied:
                    raise  # re-raise original 404
            else:
                raise
    else:
        # ...existing code for uploaded file branch...
        ext = os.path.splitext(file.filename)[1]
        uid = str(uuid.uuid4())
        original_path = os.path.join(UPLOAD_DIR, uid + ext)
        predicted_path = os.path.join(PREDICTED_DIR, uid + ext)
        with open(original_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        original_filename = file.filename
        predicted_filename = file.filename
        original_key = f"{user_folder}/original/{original_filename}"
        predicted_key = f"{user_folder}/predicted/{predicted_filename}"
    # Run YOLO prediction
    results = model(original_path, device="cpu")
    annotated_frame = results[0].plot()
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    # Persist to DB with local paths (unchanged logic)
    repository.save_prediction_session(uid, original_path, predicted_path, username, db)

    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        repository.save_detection_object(uid, label, score, bbox, db)
        detected_labels.append(label)

    # Upload to S3 (if configured)
    if AWS_S3_BUCKET and s3_client:
        try:
            # If using img query param we keep provided name for keys
            if img:
                original_key = f"{user_folder}/original/{img}"
                predicted_key = f"{user_folder}/predicted/{img}"
            _upload_to_s3(original_path, original_key)
            _upload_to_s3(predicted_path, predicted_key)
        except HTTPException:
            # Allow prediction to succeed even if S3 upload fails: could log here
            pass

    processing_time = round(time.time() - start_time, 2)

    return {
        "prediction_uid": uid,
        "detection_count": len(results[0].boxes),
        "labels": detected_labels,
        "time_took": processing_time,
        "s3_original_key": original_key if AWS_S3_BUCKET else None,
        "s3_predicted_key": predicted_key if AWS_S3_BUCKET else None
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

if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080,reload=True)