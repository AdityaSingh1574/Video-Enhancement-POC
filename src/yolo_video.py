import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def yolo_video_bounding_create(input_video_path):
    output_video_path = "output_video.mp4"
    cap = cv2.VideoCapture(input_video_path)
    xmin_final, ymin_final = (0,0)
    xmax_final, ymax_final = (0,0)
    max_area = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out = cv2.VideoWriter(output_video_path, 0, fps, (frame_width, frame_height))

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame, conf=0.4)

        # Get bounding boxes of all detected objects
        all_boxes = []
        for box in results[0].boxes.xyxy:
            all_boxes.append(box)
        all_boxes = np.array(all_boxes)

        # Calculate the outermost bounding box
        if len(all_boxes) > 0:
            xmin = np.min(all_boxes[:, 0])
            ymin = np.min(all_boxes[:, 1])
            xmax = np.max(all_boxes[:, 2])
            ymax = np.max(all_boxes[:, 3])
            # Draw the outermost bounding box
            width = xmax - xmin
            height = ymax - ymin
            if max_area < int(width * height):
                xmin_final = xmin
                ymin_final = ymin
                xmax_final = xmax
                ymax_final = ymax
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

        # Write the frame with the outermost bounding box to the output video
        out.write(frame)

    # Release video capture and writer objects
    cap.release()
    out.release()
    print(xmin_final, ymin_final, xmax_final, ymax_final)
    print("VIDEO SAVED HERE ------", output_video_path)
    return output_video_path
