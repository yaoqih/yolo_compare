from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolo11n.pt")
print(model)
# Display model information (optional)
model.info()

# # Train the model on the COCO8 example dataset for 100 epochs

results = model.train(data="fruit.yaml", epochs=100, imgsz=640,device=[0],workers=12,batch=64)

# # Run inference with the YOLOv8n model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")