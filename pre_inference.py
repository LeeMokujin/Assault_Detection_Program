from ultralytics import YOLO

def main():
    # Load the trained model
    model = YOLO('runs/detect/pretraining_train/weights/best.pt')  # Change this to your trained model path

    # Predict on an image
    results = model("https://ultralytics.com/images/bus.jpg")
    
    # Display the results (optional)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
