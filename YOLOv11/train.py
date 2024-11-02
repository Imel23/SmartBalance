from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/home/imad/Desktop/SmartBalance/Dataset/Comida1.0.v4i.yolov11/data.yaml", epochs=100, imgsz=640)


