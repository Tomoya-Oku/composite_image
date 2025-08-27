"""
    File: composite_image_extended.py
    Author: renyunfan
    Email: renyf@connect.hku.hk
    Description: [ A python script to create a composite image from a video.]
    All Rights Reserved 2023

    Updated by Yicheng Chen in 08/2025: 
    - add transparency control for moving objects
    - enhanced to always keep first and last frame
    - replaced skip_frame with snap_num for better control
"""

import cv2
import numpy as np
from enum import Enum
import argparse


class CompositeMode(Enum):
    MAX_VARIATION = 0
    MIN_VALUE = 1
    MAX_VALUE = 2


class CompositeImage:

    def __init__(self, mode, video_path, start_t = 0, end_t = 999, snap_num = 10, alpha = 1.0):
        self.video_path = video_path
        self.snap_num = snap_num
        self.start_t = start_t
        self.end_t = end_t
        self.mode = mode
        self.alpha = alpha

    def max_variation_update(self, image):
        delta_img = image - self.ave_img
        image_norm = np.linalg.norm(image, axis=2)
        delta_norm = image_norm - self.ave_img_norm
        abs_delta_norm = np.abs(delta_norm)
        delta_mask = abs_delta_norm > self.abs_diff_norm
        diff_mask = abs_delta_norm <= self.abs_diff_norm
        delta_mask = np.stack((delta_mask.T, delta_mask.T, delta_mask.T)).T.astype(np.float32)
        diff_mask = np.stack((diff_mask.T, diff_mask.T, diff_mask.T)).T.astype(np.float32)
        # Apply transparency control
        self.diff_img = self.diff_img * diff_mask + delta_img * delta_mask * self.alpha
        self. diff_norm = np.linalg.norm(self.diff_img, axis=2)
        self.abs_diff_norm = np.abs(self.diff_norm)

    def min_value_update(self, image):
        image_norm = np.linalg.norm(image, axis=2)
        cur_min_image = self.diff_img + self.ave_img
        cur_min_image_norm = np.linalg.norm(cur_min_image,axis=2)
        delta_mask = cur_min_image_norm > image_norm
        min_mask = cur_min_image_norm <= image_norm
        delta_mask = np.stack((delta_mask.T, delta_mask.T, delta_mask.T)).T.astype(np.float32)
        min_mask = np.stack((min_mask.T, min_mask.T, min_mask.T)).T.astype(np.float32)
        new_min_img = image * delta_mask + min_mask * cur_min_image
        # Apply transparency control
        diff_contribution = new_min_img - self.ave_img
        self.diff_img = diff_contribution * self.alpha

    import numpy as np

    def max_value_update(self, image):
        image_norm = np.linalg.norm(image, axis=2)
        cur_min_image = self.diff_img + self.ave_img
        cur_min_image_norm = np.linalg.norm(cur_min_image, axis=2)
        delta_mask = cur_min_image_norm < image_norm
        min_mask = cur_min_image_norm >= image_norm
        delta_mask = np.stack((delta_mask, delta_mask, delta_mask), axis=2).astype(np.float32)
        min_mask = np.stack((min_mask, min_mask, min_mask), axis=2).astype(np.float32)
        new_min_img = image * delta_mask + min_mask * cur_min_image
        # Apply transparency control
        diff_contribution = new_min_img - self.ave_img
        self.diff_img = diff_contribution * self.alpha

    def extract_frames(self):
        # Open video file
        if(self.video_path == None):
            return None
        video = cv2.VideoCapture(self.video_path)
        # Get video frame rate
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate start and end frame indices
        start_frame = int(self.start_t * fps)
        end_frame = min(int(self.end_t * fps), total_frames - 1)
        
        # Calculate total frames in the specified range
        total_range_frames = end_frame - start_frame + 1
        
        # Ensure snap_num doesn't exceed available frames
        actual_snap_num = min(self.snap_num, total_range_frames)
        
        imgs = []
        
        if actual_snap_num <= 1:
            # If only 1 frame requested, take the first frame
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = video.read()
            if ret:
                imgs.append(frame)
                print(f"Added single frame at position {start_frame}")
        elif actual_snap_num == 2:
            # If 2 frames requested, take first and last
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, first_frame = video.read()
            if ret:
                imgs.append(first_frame)
                print(f"Added first frame at position {start_frame}")
            
            if end_frame > start_frame:
                video.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
                ret, last_frame = video.read()
                if ret:
                    imgs.append(last_frame)
                    print(f"Added last frame at position {end_frame}")
        else:
            # For 3 or more frames, distribute evenly including first and last
            frame_positions = []
            
            # Always include first frame
            frame_positions.append(start_frame)
            
            # Calculate intermediate frame positions
            if actual_snap_num > 2:
                # Distribute intermediate frames evenly
                for i in range(1, actual_snap_num - 1):
                    position = start_frame + int((end_frame - start_frame) * i / (actual_snap_num - 1))
                    frame_positions.append(position)
            
            # Always include last frame (if different from first)
            if end_frame > start_frame:
                frame_positions.append(end_frame)
            
            # Remove duplicates and sort
            frame_positions = sorted(list(set(frame_positions)))
            
            # Extract frames at calculated positions
            for pos in frame_positions:
                video.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = video.read()
                if ret:
                    imgs.append(frame)
                    print(f"Added frame at position {pos}")

        # Release video object
        video.release()
        print(f"Total frames extracted: {len(imgs)} out of {actual_snap_num} requested")
        print(f"Frame range: {start_frame} to {end_frame} ({total_range_frames} total frames available)")
        return imgs

    def merge_images(self):
        image_files = self.extract_frames()
        if(image_files == None or len(image_files) < 1):
            print("Error: no image extracted, input video path at: ", self.video_path)
            exit(1)
        first_image = image_files[0]
        height, width, _ = first_image.shape

        # Iterate through each image, take maximum pixel values and composite them onto the blank canvas
        sum_image = np.zeros((height, width, 3), dtype=np.float32)
        img_num = len(image_files)

        for image_file in image_files:
            image = image_file.astype(np.float32)
            sum_image += image

        self.ave_img = sum_image / img_num

        self.ave_img_norm = np.linalg.norm(self.ave_img, axis=2)
        self.diff_norm = np.zeros((height, width), dtype=np.float32)
        self.abs_diff_norm = np.zeros((height, width), dtype=np.float32)
        self.diff_img = np.zeros((height, width, 3), dtype=np.float32)


        cnt = 0
        for image_file in image_files:
            cnt = cnt + 1
            print("Processing ", cnt, " / ", img_num)
            image = image_file.astype(np.float32)
            if(self.mode == CompositeMode.MAX_VARIATION):
                self.max_variation_update(image)
            elif(self.mode == CompositeMode.MIN_VALUE):
                self.min_value_update(image)
            elif(self.mode == CompositeMode.MAX_VALUE):
                self.max_value_update(image)

        merged_image = self.ave_img + self.diff_img
        merged_image = merged_image.astype(np.uint8)
        return merged_image


parser = argparse.ArgumentParser(
                    prog='CompositeImageExtended',
                    description='Convert video to composite image with enhanced frame selection.',
                    epilog='-')
parser.add_argument('--video_path', type=str, help='path of input video file.')
parser.add_argument('--mode', default='VAR', choices=['VAR', 'MAX', 'MIN'], help='mode of composite image.')
parser.add_argument('--start_t', default=0, type=float, help='start time of composite image.')
parser.add_argument('--end_t',default=999999, type=float, help='end time of composite image.')
parser.add_argument('--snap_num', default=10, type=int, help='total number of frames to merge (including first and last frames).')
parser.add_argument('--alpha', default=1.0, type=float, help='transparency of moving objects (0.0-1.0).')

args = parser.parse_args()

# Read command line arguments
path = args.video_path
mode = args.mode
start_t = args.start_t
end_t = args.end_t
snap_num = args.snap_num
alpha = args.alpha

print(" -- Load Param: video path", path)
print(" -- Load Param: mode", mode)
print(" -- Load Param: start_t", start_t)
print(" -- Load Param: end_t", end_t)
print(" -- Load Param: snap_num", snap_num)
print(" -- Load Param: alpha", alpha)

if(mode == 'MAX'):
    mode = CompositeMode.MAX_VALUE
elif(mode == 'MIN'):
    mode = CompositeMode.MIN_VALUE
elif(mode == 'VAR'):
    mode = CompositeMode.MAX_VARIATION

# Validate alpha parameter range
if alpha < 0.0 or alpha > 1.0:
    print("Error: alpha must be between 0.0 and 1.0")
    exit(1)

merger = CompositeImage(mode, path, start_t, end_t, snap_num, alpha)
merged_image = merger.merge_images()
cv2.imwrite('composite_image.jpg', merged_image)
