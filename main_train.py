from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data='/Assault_Detection_Program/ADP_dataset2/ADP_dataset2.yaml', epochs=100, imgsz=640)

    # Evaluate model performance on the validation set
    metrics = model.val()
    print(metrics)  # You can log this or save it as needed

    # result Automatically saved in file

    # Optional: export the model to ONNX format
    path = model.export(format="onnx")
    print(f"Model exported to: {path}")

if __name__ == "__main__":
    main()
