# """
# FastAPI YOLOv8 Moderation API with JSON response and video frame detection support
# """
# import os
# import uuid
# import random
# from datetime import datetime
# from typing import List

# from fastapi import FastAPI, APIRouter, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# import cv2
# import torch
# from ultralytics import YOLO

# # ------------------------------------------------------------
# # 1) Config
# # ------------------------------------------------------------
# UNSAFE_LABELS = {"Adults", "Gambling", "Political", "Violence"}
# SAFE_LABELS = [
#     "Cultural", "Entertainment", "Environment",
#     "Products", "Social", "Sports", "Technology"
# ]

# YOLO_MODEL_PATH = r"D:\PythonWorks\AllData\AungKhant\VideoImageDetect\multi-media-moderation-api\models\yolooct23epoch50update.pt"
# CONFIDENCE_THRESHOLD = 0.5
# IOU_THRESHOLD = 0.45
# IMG_SIZE = 640
# TEMP_DIR = "temp_media"
# os.makedirs(TEMP_DIR, exist_ok=True)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ------------------------------------------------------------
# # 2) Load YOLO model once
# # ------------------------------------------------------------
# try:
#     model = YOLO(YOLO_MODEL_PATH)
# except Exception as e:
#     raise RuntimeError(f"Failed to load YOLO model: {e}")

# # ------------------------------------------------------------
# # 3) FastAPI setup
# # ------------------------------------------------------------
# app = FastAPI(title="YOLOv8 Moderation API (Image + Video)")
# router = APIRouter()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ------------------------------------------------------------
# # 4) JSON Response Helper
# # ------------------------------------------------------------
# def build_response(data, message="OK", error=0, status_code=200):
#     payload = {
#         "error": error,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "message": message,
#         "data": data,
#     }
#     return JSONResponse(content=payload, status_code=status_code)

# # ------------------------------------------------------------
# # 5) Helper: Run YOLO detection on one frame/image
# # ------------------------------------------------------------
# def detect_frame(frame_bgr):
#     results = model.predict(
#         frame_bgr,
#         conf=CONFIDENCE_THRESHOLD,
#         iou=IOU_THRESHOLD,
#         device=DEVICE,
#         imgsz=IMG_SIZE,
#         verbose=False
#     )

#     res = results[0]
#     boxes = res.boxes
#     dets = []
#     unsafe_found = False

#     xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
#     cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []
#     conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []

#     # Debug print
#     print("xyxy array:", xyxy)
#     print("Number of detections:", len(xyxy))

#     if len(xyxy) == 0:
#         fake_label = random.choice(SAFE_LABELS)
#         dets.append({
#             "class": fake_label,
#             "confidence": 0.6,
#             "auto": True,
#             "xyxy": []  # empty for no detection
#         })
#         is_safe = True
#         print("No detections, fake label added:", fake_label)
#     else:
#         for (cls_idx, conf_score, box_coords) in zip(cls_arr, conf_arr, xyxy):
#             class_name = res.names.get(int(cls_idx), str(int(cls_idx)))
#             confidence = round(float(conf_score), 3)
#             dets.append({
#                 "class": class_name,
#                 "confidence": confidence,
#                 "auto": False,
#                 "xyxy": box_coords.tolist()  # convert to list for JSON
#             })
#             print(f"Detection: {class_name}, Confidence: {confidence}, Box: {box_coords}")
#             if class_name in UNSAFE_LABELS:
#                 unsafe_found = True
#         is_safe = not unsafe_found
#         print("Is safe:", is_safe)

#     labels = list({d["class"] for d in dets})
#     return {
#         "is_safe": is_safe,
#         "content_type": labels,
#         "detections": dets,
#         "xyxy": xyxy.tolist() if len(xyxy) else []
#     }

# # ------------------------------------------------------------
# # 6) Detect in single image (path or upload)
# # ------------------------------------------------------------
# def process_image_file(path: str):
#     img = cv2.imread(path)
#     result = detect_frame(img)
#     return {
#         "filename": os.path.basename(path),
#         "is_safe": result["is_safe"],
#         "content_type": result["content_type"],
#         "detections": result["detections"],
#         "xyxy": result["xyxy"]
#     }

# # ------------------------------------------------------------
# # 7) Detect frames in a video (detailed version)
# # ------------------------------------------------------------
# def process_video_file(video_path: str):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open video")

#     fps = cap.get(cv2.CAP_PROP_FPS) or 25
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
#     duration_sec = frame_count / fps if frame_count else 0

#     # set sampling rate
#     if duration_sec < 60:
#         interval_sec = 3
#     elif duration_sec < 120:
#         interval_sec = 5
#     else:
#         interval_sec = 7

#     interval_frames = max(1, int(fps * interval_sec))
#     frame_idx = 0
#     results = []
#     unsafe_overall = False

#     while True:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         ret, frame = cap.read()
#         if not ret:
#             break

#         det = detect_frame(frame)
#         frame_detections = [
#             {
#                 "class": d["class"],
#                 "confidence": d["confidence"],
#                 "xyxy": d.get("xyxy", [])
#             }
#             for d in det["detections"]
#         ]
#         unsafe_in_frame = any(d["class"] in UNSAFE_LABELS for d in det["detections"])
#         if unsafe_in_frame:
#             unsafe_overall = True

#         results.append({
#             "frame_index": int(frame_idx),
#             "time_sec": round(frame_idx / fps, 2),
#             "is_safe": det["is_safe"],
#             "detected_classes": [d["class"] for d in det["detections"]],
#             "detections": frame_detections,
#             "xyxy": det["xyxy"]
#         })

#         frame_idx += interval_frames
#         if frame_count and frame_idx >= frame_count:
#             break

#     cap.release()
#     overall_safe = not unsafe_overall

#     return {
#         "video_name": os.path.basename(video_path),
#         "frames_analyzed": len(results),
#         "fps": round(fps, 2),
#         "duration_sec": round(duration_sec, 2),
#         "is_safe": overall_safe,
#         "results": results
#     }

# # ------------------------------------------------------------
# # 8) Pydantic schema
# # ------------------------------------------------------------
# class ImageInput(BaseModel):
#     images: List[str]

# # ------------------------------------------------------------
# # 9) Endpoints
# # ------------------------------------------------------------
# @router.get("/health")
# def health():
#     return build_response(data=[], message="healthy", error=0)

# @router.post("/image-detect")
# async def image_detect(input_data: ImageInput):
#     try:
#         results = []
#         for img_path in input_data.images:
#             if not os.path.exists(img_path):
#                 results.append({"image": img_path, "error": "File not found"})
#                 continue
#             results.append(process_image_file(img_path))
#         return build_response(results, "Processed images")
#     except Exception as e:
#         return build_response([], f"Image detection failed: {e}", 1, 500)

# @router.post("/upload-image")
# async def upload_and_detect_images(files: List[UploadFile] = File(...)):
#     try:
#         results = []
#         for file in files:
#             temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}_{file.filename}")
#             with open(temp_path, "wb") as f:
#                 f.write(await file.read())
#             res = process_image_file(temp_path)
#             results.append(res)
#             os.remove(temp_path)
#         return build_response(results, "Processed uploaded images")
#     except Exception as e:
#         return build_response([], f"Upload image detection failed: {e}", 1, 500)

# @router.post("/video-detect")
# async def video_detect(file: UploadFile = File(...)):
#     try:
#         if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
#             return build_response([], "Unsupported file type", 1, 400)

#         temp_path = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}_{file.filename}")
#         with open(temp_path, "wb") as f:
#             f.write(await file.read())

#         result = process_video_file(temp_path)
#         os.remove(temp_path)
#         return build_response([result], "Processed video successfully")
#     except Exception as e:
#         return build_response([], f"Video detection failed: {e}", 1, 500)

# # ------------------------------------------------------------
# # 10) Register + run
# # ------------------------------------------------------------
# app.include_router(router)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("yolo_image_video_detection:app", host="0.0.0.0", port=8000, reload=False)


import sys
import types

if "imghdr" not in sys.modules:
    fake_imghdr = types.ModuleType("imghdr")
    fake_imghdr.what = lambda *args, **kwargs: None
    sys.modules["imghdr"] = fake_imghdr

    
"""
FastAPI YOLOv8 Moderation API with JSON response and video frame detection support
"""
import os
import io
import uuid
import random
import shutil
from datetime import datetime
from typing import List

from fastapi import FastAPI, APIRouter, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import cv2
import torch
from ultralytics import YOLO

# ------------------------------------------------------------
# 1) Config
# ------------------------------------------------------------
UNSAFE_LABELS = {"Adults", "Gambling", "Political", "Violence"}
SAFE_LABELS = [
    "Cultural", "Entertainment", "Environment",
    "Products", "Social", "Sports", "Technology"
]

YOLO_MODEL_PATH = r"D:\PythonWorks\AllData\AungKhant\VideoImageDetect\multi-media-moderation-api\models\yolooct27epoch100update.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
TEMP_DIR = "temp_media"
os.makedirs(TEMP_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# 2) Load YOLO model once
# ------------------------------------------------------------
try:
    model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

# ------------------------------------------------------------
# 3) FastAPI setup
# ------------------------------------------------------------
app = FastAPI(title="YOLOv8 Moderation API (Image + Video)")
router = APIRouter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# 4) JSON Response Helper
# ------------------------------------------------------------
def build_response(data, message="OK", error=0, status_code=200):
    payload = {
        "error": error,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
        "data": data,
    }
    return JSONResponse(content=payload, status_code=status_code)

# ------------------------------------------------------------
# 5) Helper: Run YOLO detection on one frame/image
# ------------------------------------------------------------
def detect_frame(frame_bgr):
    results = model.predict(
        frame_bgr,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        imgsz=IMG_SIZE,
        verbose=False
    )

    res = results[0]
    boxes = res.boxes
    dets = []
    unsafe_found = False

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
    cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []
    conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []

    if len(xyxy) == 0:
        fake_label = random.choice(SAFE_LABELS)
        dets.append({"class": fake_label, "confidence": 0.6, "auto": True})
        is_safe = True
    else:
        for (cls_idx, conf_score) in zip(cls_arr, conf_arr):
            class_name = res.names.get(int(cls_idx), str(int(cls_idx)))
            confidence = round(float(conf_score), 3)
            dets.append({
                "class": class_name,
                "confidence": confidence,
                "auto": False,
            })
            if class_name in UNSAFE_LABELS:
                unsafe_found = True
        is_safe = not unsafe_found

    labels = list({d["class"] for d in dets})
    return {"is_safe": is_safe, "content_type": labels, "detections": dets}

# ------------------------------------------------------------
# 6) Detect in single image (path or upload)
# ------------------------------------------------------------
def process_image_file(path: str):
    img = cv2.imread(path)
    result = detect_frame(img)
    return {
        "filename": os.path.basename(path),
        "is_safe": result["is_safe"],
        "content_type": result["content_type"],
        "detections": result["detections"],
    }

# ------------------------------------------------------------
# 7) Detect frames in a video (detailed version)
# ------------------------------------------------------------
def process_video_file(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_sec = frame_count / fps if frame_count else 0

    # set sampling rate (every 10–20 sec)
    if duration_sec < 60:
        interval_sec = 3
    elif duration_sec < 120:
        interval_sec = 5
    else:
        interval_sec = 7

    interval_frames = max(1, int(fps * interval_sec))
    frame_idx = 0
    results = []
    unsafe_overall = False

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        det = detect_frame(frame)
        frame_detections = [
            {"class": d["class"], "confidence": d["confidence"]}
            for d in det["detections"]
        ]
        unsafe_in_frame = any(d["class"] in UNSAFE_LABELS for d in det["detections"])
        if unsafe_in_frame:
            unsafe_overall = True

        results.append({
            "frame_index": int(frame_idx),
            "time_sec": round(frame_idx / fps, 2),
            "is_safe": det["is_safe"],
            "detected_classes": [d["class"] for d in det["detections"]],
            "detections": frame_detections
        })

        frame_idx += interval_frames
        if frame_count and frame_idx >= frame_count:
            break

    cap.release()

    overall_safe = not unsafe_overall
    return {
        "video_name": os.path.basename(video_path),
        "frames_analyzed": len(results),
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 2),
        "is_safe": overall_safe,
        "results": results,
    }

# ------------------------------------------------------------
# 8) Pydantic schema
# ------------------------------------------------------------
class ImageInput(BaseModel):
    images: List[str]

# ------------------------------------------------------------
# 9) Endpoints
# ------------------------------------------------------------
@router.get("/health")
def health():
    return build_response(data=[], message="healthy", error=0)

@router.post("/image-detect")
async def image_detect(input_data: ImageInput):
    try:
        results = []
        for img_path in input_data.images:
            if not os.path.exists(img_path):
                results.append({"image": img_path, "error": "File not found"})
                continue
            results.append(process_image_file(img_path))
        return build_response(results, "Processed images")
    except Exception as e:
        return build_response([], f"Image detection failed: {e}", 1, 500)

@router.post("/upload-image")
async def upload_and_detect_images(files: List[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}_{file.filename}")
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            res = process_image_file(temp_path)
            results.append(res)
            os.remove(temp_path)
        return build_response(results, "Processed uploaded images")
    except Exception as e:
        return build_response([], f"Upload image detection failed: {e}", 1, 500)

@router.post("/video-detect")
async def video_detect(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
            return build_response([], "Unsupported file type", 1, 400)

        temp_path = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}_{file.filename}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        result = process_video_file(temp_path)
        os.remove(temp_path)
        return build_response([result], "Processed video successfully")
    except Exception as e:
        return build_response([], f"Video detection failed: {e}", 1, 500)

# ------------------------------------------------------------
# 10) Register + run
# ------------------------------------------------------------
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("yolo_image_video_detection:app", host="0.0.0.0", port=8000, reload=False)