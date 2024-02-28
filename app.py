import streamlit as st
import yolo_video
import video_crop
def main():
     st.title("Object Detection with YOLO")
     st.write("Upload a video file to detect objects.")

     uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "wmv", "mkv"])

     if uploaded_file is not None:
          if st.button("Process Video"):
               temp_video_path = "temp_video.mp4"
               with open(temp_video_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
               print('hehehehe',temp_video_path)
               temp_video_path_2 = video_crop.get_contour_remove_black_borders(temp_video_path)
               print('hahahaha',temp_video_path_2)
               output_video_path = video_crop.zoomed_to_fill_result_video(temp_video_path_2)
               # output_video_path = yolo_video.yolo_video_bounding_create(temp_video_path)
               
               print("Printing output video path:\n",output_video_path)

               st.write("Processed Video:")
               processed_video_file = open(output_video_path, "rb").read()
               st.video(processed_video_file)

               import os
               os.remove(temp_video_path)
               # os.remove(output_video_path)

if __name__ == "__main__":
    main()