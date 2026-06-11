from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset_split",
    epochs=20,
    imgsz=224,
    batch=4,
    workers=0,
    name="agropest_final"
)