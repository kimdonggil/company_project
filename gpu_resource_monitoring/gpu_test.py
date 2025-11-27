from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="coco8.yaml", epochs=50, imgsz=640, device=[-1])
