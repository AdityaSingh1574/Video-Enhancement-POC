import cv2
import numpy as np
import random
from ultralytics import YOLO
import streamlit as st

model = YOLO("yolov8n.pt")


def crop_video(input_file, x1, y1, x2, y2):
  output_file = "output_file.mp4"
  # Open the input video
  cap = cv2.VideoCapture(input_file)

  # Get video properties
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # Define the codec and create video writer object
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or other suitable codec
  out = cv2.VideoWriter(output_file, fourcc, fps, (x2 - x1, y2 - y1))

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # Crop the frame
    cropped_frame = frame[y1:y2, x1:x2]

    # Write the cropped frame to the output video
    out.write(cropped_frame)

  # Release resources
  cap.release()
  out.release()
  cv2.destroyAllWindows()
  return output_file


def yolo_video_bounding(input_video_path):
  # output_video_path = "result_video/output_video1.mp4"
  cap = cv2.VideoCapture(input_video_path)

  # fps = int(cap.get(cv2.CAP_PROP_FPS))
  # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # Define the codec and create VideoWriter object
  # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  # out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
  xmin_final, ymin_final = (0, 0)
  xmax_final, ymax_final = (0, 0)
  max_area = 0
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
      # cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
      width = xmax - xmin
      height = ymax - ymin
      if max_area < int(width * height):
        xmin_final = xmin
        ymin_final = ymin
        xmax_final = xmax
        ymax_final = ymax
    # Write the frame with the outermost bounding box to the output video
    # out.write(frame)

  # Release video capture and writer objects
  cap.release()
  # print(xmin_final, ymin_final, xmax_final, ymax_final)
  return (int(xmin_final), int(ymin_final), int(xmax_final), int(ymax_final))


def get_random_frame(video_path):

  # Open the video capture object
  cap = cv2.VideoCapture(video_path)

  # Check if video opened successfully
  if not cap.isOpened():
    print("Error opening video file!")
    return None

  # Get video length in frames
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  # Generate a random timestamp within video duration
  random_timestamp = random.randint(0, frame_count - 1)

  # Set the video capture position to the random timestamp
  cap.set(cv2.CAP_PROP_POS_FRAMES, random_timestamp)

  # Read the frame
  ret, frame = cap.read()

  # Release the video capture object
  cap.release()

  # Return the extracted frame if successful
  if ret:
    return frame
  else:
    print("Failed to read frame!")
    return None


def get_contour_remove_black_borders(vid_link):
  # Open the video file
  cap = cv2.VideoCapture(vid_link)

  if not cap.isOpened():
    print("Error: Could not open video.")
    return

  # Read the first frame to get video dimensions
  ret, frame = cap.read()
  if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    return

  x, y, w, h = find_most_common_of_largest_contours_test(vid_link)

  # Draw the rectangle on the original frame
  result_frame = frame.copy()
  cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

  # Display the result
  # cv2.imshow("Black Borders Detection in ", result_frame)
  # cv2.waitKey(0)
  cv2.destroyAllWindows()

  output_file = "temp_video_2.mp4"
  # Save the cropped frame to a new video file
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (adjust as needed)
  out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

  while True:
    # print("Reading the video \n")
    ret, frame = cap.read()
    if not ret:
      break
    cropped_frame = frame[y:y + h, x:x + w]
    out.write(cropped_frame)

  if out.isOpened():
    out.write(cropped_frame)
    print(f"Cropped video saved to {output_file}")
    out.release()
  else:
    print("Error: Could not initialize VideoWriter. get_contour_remove_black_borders")
  return output_file



def detect_black_borders(vid_link):
  # Open the video file
  cap = cv2.VideoCapture(vid_link)

  if not cap.isOpened():
    print("Error: Could not open video.")
    return


  # Read the frame at the specific time
  ret, frame = cap.read()


  # Convert the frame to grayscale
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # # Calculate the black borders
  _, threshold = cv2.threshold(gray_frame, 10, 255, cv2.THRESH_BINARY )

  # cv2.imshow("thresold imgae", threshold)

  # Find contours on the smoothed image
  contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  # for i,con in enumerate(contours):
  #   x, y, w, h = cv2.boundingRect(con)
  #
  #   frame_copy = frame.copy()
  #
  #   cv2.rectangle(frame_copy,(x,y), (x + w,y + h), (0,255,0), 2)


    # cv2.imshow(f"contour number : {i}", frame_copy)
    # cv2.waitKey(0)

  # Find the largest contour (black border)
  if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    center, radius = cv2.minEnclosingCircle(largest_contour)

    # Distance to the nearest point on the contour is simply the radius


    # print(nearest_distance)
    # Approximate the contour with a polygon
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)  # Adjust epsilon as needed
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Get the bounding rectangle around the contour
    x, y, w, h = cv2.boundingRect(approx_contour)

    nearest_distance = h - radius

    if nearest_distance < 0:
      nearest_distance = 0
    else:
      nearest_distance = int(nearest_distance)

    # Filter out faded borders by applying additional thresholding
    cropped_frame = frame[y:y + h - 2 * nearest_distance - 1, x:x + w]
    cropped_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(cropped_gray, 10, 255, cv2.THRESH_BINARY)

    # Use the mask to further refine the cropped frame
    refined_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)

    # cv2.imshow("Refined Result", refined_frame)
    # cv2.waitKey(0)



    while True:
      # Read the next frame
      ret, frame = cap.read()

      if not ret:
        break

      # Draw the rectangle on the current frame
      cv2.rectangle(frame, (x, y), (x + w, y + h - 2 * nearest_distance - 1), (0, 255, 0), 2)

      # Display the frame with the rectangle
      # cv2.imshow("Cropped Video with Rectangle", frame)

      # Break the loop if 'q' key is pressed
      if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

  else:
    print("No black borders detected.")



def smart_zoom_to_fill(input_file, frame_height, frame_width, subject_bbox): # subject_bbox = [y1,x1,h,w]
  subject_top_left_x = subject_bbox[1]
  subject_top_left_y = subject_bbox[0]


  subject_height = subject_bbox[2]

  subject_width = subject_bbox[3]


  print(f"frame_height : {type(frame_height)} \nframe_height - (subject_top_left_y + subject_height) : {type(frame_height - (subject_top_left_y + subject_height))}, \nsubject_top_left_y : {type(subject_top_left_y)}")


  min_dis_ver = min(frame_height - (subject_top_left_y + subject_height), subject_top_left_y)



  min_dis_hor = min(subject_top_left_x, frame_width - (subject_top_left_x + subject_width))

  min_dis_hor = min(min_dis_ver , min_dis_hor)

  new_height = frame_height - 2 * min_dis_ver
  new_width = frame_width - 2 * min_dis_hor

  print(min_dis_hor, min_dis_ver)

  # Calculate the top-left and bottom-right corners of the zoomed region
  top_left_x = min_dis_hor
  top_left_y = min_dis_ver
  bottom_right_x = top_left_x + new_width
  bottom_right_y = top_left_y + new_height

  

  output_file = crop_video(input_file, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
  return output_file

@st.cache_data
def zoomed_to_fill_result_video(vid_link):
  cap = cv2.VideoCapture(vid_link)

  if not cap.isOpened():
    print("Error: Could not open video. in zoomed_to_fill_result_video")
    return

  # Read the frame at the specific time
  ret, frame = cap.read()

  frame_height, frame_width = frame.shape[:2]

  x1,y1,x2,y2 = yolo_video_bounding(vid_link)
  print(x1, x2, y2, y1)
  output_path = smart_zoom_to_fill(vid_link, frame_height, frame_width, [y1, x1, y2-y1, x2-x1])

  cap.release()
  cv2.destroyAllWindows()
  return output_path


def is_approximated_rectangle(contour):
  # Approximate the contour to a polygon (rectangle)
  epsilon = 0.01 * cv2.arcLength(contour, True)  # Adjust epsilon as needed
  approx_contour = cv2.approxPolyDP(contour, epsilon, True)

  # Check if the approximated contour has 4 vertices (rectangle)
  return len(approx_contour) >= 4

def find_most_common_of_largest_contours_test(video_path, skip_time=2):
  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    print("Error: Could not open video in find_most_common_of_largest_contours_test")
    return

  largest_contour_counts = dict()  # Dictionary to store the count of each largest contour

  frame_count = 0

  while True:
    ret, frame = cap.read()

    if not ret:
      break

    # Increment frame count
    frame_count += 1

    if frame_count % skip_time != 0:
      continue

    # Skip frames not in the sample


    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresholded_frame = cv2.threshold(gray_frame, 5, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
      # Find the largest contour
      current_largest_contour = max(contours, key=cv2.contourArea)


      if not is_approximated_rectangle(current_largest_contour):
        continue

      # Approximate the contour with a polygon
      epsilon = 0.05 * cv2.arcLength(current_largest_contour, True)  # Adjust epsilon as needed
      approx_contour = cv2.approxPolyDP(current_largest_contour, epsilon, True)

      contour_string = approx_contour.tostring()

      if contour_string not in largest_contour_counts:
        shape_of_contour = approx_contour.shape
        dtype_of_contour = approx_contour.dtype
        largest_contour_counts[contour_string] = [1,shape_of_contour,dtype_of_contour]
      else:
        count = largest_contour_counts[contour_string][0]
        largest_contour_counts[contour_string]  = [count + 1, largest_contour_counts[contour_string][1], largest_contour_counts[contour_string][2]]

      # Draw the largest contour on the original frame
      frame_with_largest_contour = frame.copy()
      cv2.drawContours(frame_with_largest_contour, [approx_contour], -1, (0, 255, 0), 2)

      # Display frames with contours
      # cv2.imshow("Frame with largest Rectangle Contour", frame_with_largest_contour)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break



  largest_contour = None
  max_count = 0
  for contour,value in largest_contour_counts.items():
    if max_count < value[0]:
      max_count = value[0]
      largest_contour = np.frombuffer(contour, dtype=value[2]).reshape(value[1])



  l_x,l_y,l_w,l_h = cv2.boundingRect(largest_contour)

  avg_contour_params = [[l_x,l_y,l_w,l_h]]
  largest_area = l_h * l_w

  for contour,value in largest_contour_counts.items():

    current_contour =  np.frombuffer(contour, dtype=value[2]).reshape(value[1])
    temp_x,temp_y,temp_w,temp_h = cv2.boundingRect(current_contour)
    current_area = temp_w * temp_h

    rel_diff = int(((largest_area - current_area) / largest_area) * 100)

    # setting the minn difference to be
    if rel_diff < 5:
      avg_contour_params.append([temp_x,temp_y,temp_w,temp_h])

  avg_x,avg_y,avg_w,avg_h = (0,0,0,0)

  for coord in avg_contour_params:
    avg_x = avg_x + coord[0]
    avg_y = avg_y + coord[1]
    avg_w = avg_w + coord[2]
    avg_h = avg_h + coord[3]

  number_of_coords = len(avg_contour_params)


  print(f"Number of averages : {number_of_coords}")

  if number_of_coords == 1:
    avg_x = 2 * avg_x
    avg_y = 2 * avg_y
    avg_w = 2 * avg_w
    avg_h = 2 * avg_h
    number_of_coords = 2

  
  avg_x = int(avg_x // number_of_coords)
  avg_y = int(avg_y // number_of_coords)
  avg_w = int(avg_w // number_of_coords)
  avg_h = int(avg_h // number_of_coords)

  


  # cv2.waitKey(0)
  # Release the video capture object
  cap.release()
  cv2.destroyAllWindows()


  return (avg_x,avg_y,avg_w,avg_h)



# Example usage
# if __name__ == '__main__':
#   vid_link = "C:\\Users\\aditya.singh1\\Downloads\\low_light_test_again.mp4"
#   get_contour_remove_black_borders(vid_link)
#   intermediate_link = "C:\\Users\\aditya.singh1\\Downloads\\intermediate_result.mp4"
#   zoomed_to_fill_result_video(intermediate_link)