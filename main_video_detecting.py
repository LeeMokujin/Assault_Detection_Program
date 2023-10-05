import cv2
from ultralytics import YOLO

def main():
    # Load the trained model
    model = YOLO('runs/detect/train/weights/best.pt')  # Change this to your trained model path

    # Open the video using OpenCV
    cap = cv2.VideoCapture('D:\Assault_Detection_Program\ADP_video_dataset\Assault015_x264.mp4')

    while cap.isOpened():
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Predict on the current frame
        results = model(frame)

        # Retrieve bounding boxes, if they exist
        if hasattr(results, 'boxes'):
            boxes = results.boxes

            # Draw bounding boxes on the frame
            for box in boxes:
                # Get coordinates and label
                x1, y1, x2, y2 = box[:4]
                print(x1, x2, y1, y2)
                print(boxes)
                class_id = int(box[4])
                label =  results.names[class_id]

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Draw box
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Draw label

        # Display the frame with bounding boxes
        cv2.imshow('YOLO Real-time Object Detection', frame)

        # Press 'q' to exit the video window
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

