import os
import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def change_brightness(self, img, value):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v,value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def process_videos(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".mp4"):
                
                video_path = os.path.join(self.input_dir, filename)
                print(f"Processing video: {video_path}")
                
                video = cv2.VideoCapture(video_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = video.get(cv2.CAP_PROP_FPS)
                frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

                print(f"Output directory: {self.output_dir}")
                os.makedirs(self.output_dir, exist_ok=True)

                # Generate flipped and brightness adjusted videos
                print("Performing horizontal flip")
                flip_h_output = cv2.VideoWriter(os.path.join(self.output_dir, f'{os.path.splitext(filename)[0]}_flip_h.mp4'), fourcc, fps, (frame_width, frame_height))
                print("Performing vertical flip")
                flip_v_output = cv2.VideoWriter(os.path.join(self.output_dir, f'{os.path.splitext(filename)[0]}_flip_v.mp4'), fourcc, fps, (frame_width, frame_height))
                print("Performing low brightness")
                bright_low_output = cv2.VideoWriter(os.path.join(self.output_dir, f'{os.path.splitext(filename)[0]}_bright_low.mp4'), fourcc, fps, (frame_width, frame_height))
                print("Performing medium brightness")
                bright_medium_output = cv2.VideoWriter(os.path.join(self.output_dir, f'{os.path.splitext(filename)[0]}_bright_medium.mp4'), fourcc, fps, (frame_width, frame_height))
                print("Performing high brightness")
                bright_high_output = cv2.VideoWriter(os.path.join(self.output_dir, f'{os.path.splitext(filename)[0]}_bright_high.mp4'), fourcc, fps, (frame_width, frame_height))

                while True:
                    ret, frame = video.read()
                    if not ret:
                        break

                    # Flip frames horizontally and vertically
                    flip_h = cv2.flip(frame, 1)
                    flip_v = cv2.flip(frame, 0)

                    # Adjust brightness
                    bright_low = self.change_brightness(frame, -50)
                    bright_medium = self.change_brightness(frame, 0)
                    bright_high = self.change_brightness(frame, 50)

                    flip_h_output.write(flip_h)
                    flip_v_output.write(flip_v)
                    bright_low_output.write(bright_low)
                    bright_medium_output.write(bright_medium)
                    bright_high_output.write(bright_high)

                video.release()
                flip_h_output.release()
                flip_v_output.release()
                bright_low_output.release()
                bright_medium_output.release()
                bright_high_output.release()
