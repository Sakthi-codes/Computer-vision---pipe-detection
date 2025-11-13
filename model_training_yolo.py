from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model.train(data='C:/Users/sakth/Downloads/pipe_dedection/data.yaml', epochs=100, imgsz=640, device=0, cache=False, workers=0)
