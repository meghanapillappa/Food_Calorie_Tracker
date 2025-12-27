from ultralytics import YOLO

model = YOLO("best.pt")
print("CLASSES:", model.names)
print("NUMBER OF CLASSES:", len(model.names))
