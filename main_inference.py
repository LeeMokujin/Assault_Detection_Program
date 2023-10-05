from ultralytics import YOLO

def main():
    # Load the trained model
    model = YOLO('runs/detect/train/weights/best.pt')  # Change this to your trained model path

    # Predict on an image
    results = model('D:\Assault_Detection_Program\ADP_video_dataset\Assault051_x264.mp4')
    
    # Display the results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()