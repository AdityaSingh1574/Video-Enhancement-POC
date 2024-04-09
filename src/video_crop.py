import cv2
import numpy as np
import random
import streamlit as st
from ultralytics import YOLO
import streamlit as st
model = YOLO("yolov8n.pt")


def crop_video(input_file, x1, y1, x2, y2):
  output_file = "output_file.mp4"
  cap = cv2.VideoCapture(input_file)

  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  fourcc = cv2.VideoWriter_fourcc(*'avc1')
  out = cv2.VideoWriter(output_file, fourcc, fps, (x2 - x1, y2 - y1))

  print(f"Inside the crop_video() function-  "
        f"Width : {width}"
        f"Height : {height} \n")

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    cropped_frame = frame[y1:y2, x1:x2]
    out.write(cropped_frame)

  cap.release()
  out.release()
  return output_file


def yolo_video_bounding(input_video_path):
  cap = cv2.VideoCapture(input_video_path)

  # fps = int(cap.get(cv2.CAP_PROP_FPS))
  # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # Define the codec and create VideoWriter object
  # fourcc = cv2.VideoWriter_fourcc(*'avc1')
  # out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
  xmin_final, ymin_final = (0, 0)
  xmax_final, ymax_final = (0, 0)
  max_area = 0

  fps = cap.get(cv2.CAP_PROP_FPS)
  frames_to_skip = fps * 0.5

  while cap.isOpened():
    current_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    next_target_frame = current_frame_number + frames_to_skip
    cap.set(cv2.CAP_PROP_POS_FRAMES, next_target_frame)

    ret, frame = cap.read()
    if not ret:
      break

    results = model(frame, conf=0.4)

    all_boxes = []
    for box in results[0].boxes.xyxy:
      all_boxes.append(box)
    all_boxes = np.array(all_boxes)

    if len(all_boxes) > 0:
      xmin = np.min(all_boxes[:, 0])
      ymin = np.min(all_boxes[:, 1])
      xmax = np.max(all_boxes[:, 2])
      ymax = np.max(all_boxes[:, 3])

      width = xmax - xmin
      height = ymax - ymin
      if max_area < int(width * height):
        xmin_final = xmin
        ymin_final = ymin
        xmax_final = xmax
        ymax_final = ymax

  cap.release()
  return (int(xmin_final), int(ymin_final), int(xmax_final), int(ymax_final))


# def get_random_frame(video_path):
#   cap = cv2.VideoCapture(video_path)

#   # if not cap.isOpened():
#     # print("Error opening video file!")
#     # return None

#   frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#   random_timestamp = random.randint(0, frame_count - 1)

#   # Set the video capture position to the random timestamp
#   cap.set(cv2.CAP_PROP_POS_FRAMES, random_timestamp)

#   # Read the frame
#   ret, frame = cap.read()

#   # Release the video capture object
#   cap.release()

#   # Return the extracted frame if successful
#   # if ret:
#   return frame
#   # else:
#     # print("Failed to read frame!")
#     # return None


def get_contour_remove_black_borders(vid_link):
  cap = cv2.VideoCapture(vid_link)

  # if not cap.isOpened():
    # print("Error: Could not open video.")
    # return

  ret, frame = cap.read()
  # if not ret:
    # print("Error: Could not read the first frame.")
    # cap.release()
    # return

  x, y, w, h = find_most_common_of_largest_contours_test(vid_link)

  # Draw the rectangle on the original frame
  result_frame = frame.copy()

  output_file = "temp_video_2.mp4"
  fourcc = cv2.VideoWriter_fourcc(*'avc1')
  out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

  # fps = cap.get(cv2.CAP_PROP_FPS)
  # frames_to_skip = fps * 0.25  # Skip 2 seconds

  while True:
    ret, frame = cap.read()
    if not ret:
      break
    cropped_frame = frame[y:y + h, x:x + w]
    out.write(cropped_frame)

  if out.isOpened():
    # out.write(cropped_frame)
    print(f"Cropped video saved to {output_file}")
    out.release()
  else:
    print("Error: Could not initialize VideoWriter. get_contour_remove_black_borders")
  return output_file


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

  top_left_x = min_dis_hor
  top_left_y = min_dis_ver
  bottom_right_x = top_left_x + new_width
  bottom_right_y = top_left_y + new_height

  output_file = crop_video(input_file, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
  return output_file

# @st.cache_data
def zoomed_to_fill_result_video(vid_link):
  cap = cv2.VideoCapture(vid_link)

  # if not cap.isOpened():
    # print("Error: Could not open video. in zoomed_to_fill_result_video")
    # return

  # Read the frame at the specific time
  ret, frame = cap.read()

  frame_height, frame_width = frame.shape[:2]

  x1,y1,x2,y2 = yolo_video_bounding(vid_link)
  print(x1, x2, y2, y1)
  output_path = smart_zoom_to_fill(vid_link, frame_height, frame_width, [y1, x1, y2-y1, x2-x1])

  cap.release()
  # cv2.destroyAllWindows()
  return output_path


def is_approximated_rectangle(contour):
  # Approximate the contour to a polygon (rectangle)
  epsilon = 0.01 * cv2.arcLength(contour, True)  # Adjust epsilon as needed
  approx_contour = cv2.approxPolyDP(contour, epsilon, True)

  # Check if the approximated contour has 4 vertices (rectangle)
  return len(approx_contour) >= 4

def find_most_common_of_largest_contours_test(video_path, skip_time=2):
  cap = cv2.VideoCapture(video_path)

  # if not cap.isOpened():
    # print("Error: Could not open video in find_most_common_of_largest_contours_test")
    # return

  largest_contour_counts = dict()  # Dictionary to store the count of each largest contour
  frame_count = 0

  fps = cap.get(cv2.CAP_PROP_FPS)
  frames_to_skip = fps * 0.5  # Skip 2 seconds

  while True:
    ret, frame = cap.read()

    if not ret:
      break

    current_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    next_target_frame = current_frame_number + frames_to_skip
    cap.set(cv2.CAP_PROP_POS_FRAMES, next_target_frame)

    # Increment frame count
    # frame_count += 1
    #
    # if frame_count % skip_time != 0:
    #   continue

    # Skip frames not in the sample

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresholded_frame = cv2.threshold(gray_frame, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
      current_largest_contour = max(contours, key=cv2.contourArea)


      if not is_approximated_rectangle(current_largest_contour):
        continue

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
    if 0xFF == ord('q'):
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

  avg_x, avg_y, avg_w, avg_h = (0, 0, 0, 0)

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
  cap.release()
  return (avg_x, avg_y, avg_w, avg_h)