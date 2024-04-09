import streamlit as st
import os
import yolo_video
import video_crop

def main():
     st.set_page_config(layout="wide")
     st.title("StarzPlay POC")
     temp_video_path = '../temp_video.mp4'
     uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "wmv", "mkv"])
     
     placeholder = st.empty()
     with placeholder.container():
          if uploaded_file is not None:
               temp_video_path = "temp_video.mp4"
               with open(temp_video_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

               col1, col2 = st.columns(2)

               with col1:
                    st.write("Input Video:")
                    st.video(temp_video_path)

               if st.button("Process Video"):
                    with st.spinner("Processing..."):
                         # try:
                              temp_video_path_2 = video_crop.get_contour_remove_black_borders(temp_video_path)
                              output_video_path = video_crop.zoomed_to_fill_result_video(temp_video_path_2)
                              with col2:
                                   st.write("Processed Video:")
                                   processed_video_file = open(output_video_path, "rb").read()
                                   st.video(processed_video_file, format='video/mp4')
                         # except Exception as e:
                              # st.error(f"Error processing video: {e}")
     if st.button("Clear"):
          if temp_video_path is not None:
               os.remove(temp_video_path)
               os.remove('output_file.mp4')
               os.remove('temp_video_2.mp4')
               temp_video_path = None

          # if temp_video_path_2 is not None:
          #      print(temp_video_path_2)
          #      os.remove(temp_video_path_2)
          #      temp_video_path_2 = None

          # if output_video_path is not None:
          #      print(output_video_path)
          #      os.remove(output_video_path)
          #      output_video_path = None
          placeholder.empty()

               

if __name__ == "__main__":
    main()
