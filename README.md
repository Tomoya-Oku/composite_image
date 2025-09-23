# composite_image
This repo provide a python script to create a composite image from a video.

# Dependency

```bash
pip install opencv-python
```

# Example

## 1 VAR Mode (Recommend)
```bash
python composite_image.py --video_path=./example_video.mp4 --start_t=0.0 --end_t=99.0 --skip_frame=2 --mode=VAR
```
![image-20230909002327059](./img/image-20230909002327059.png)

### With Transparency Control
```bash
# Full opacity (default)
python composite_image.py --video_path=./example_video.mp4 --start_t=0.0 --end_t=99.0 --skip_frame=2 --mode=VAR --alpha=1.0

# Semi-transparent moving objects
python composite_image.py --video_path=./example_video.mp4 --start_t=0.0 --end_t=99.0 --skip_frame=2 --mode=VAR --alpha=0.5

# Very subtle moving objects
python composite_image.py --video_path=./example_video.mp4 --start_t=0.0 --end_t=99.0 --skip_frame=2 --mode=VAR --alpha=0.2
```

## 2 MIN Mode
```bash
python composite_image.py --video_path=./example_video.mp4 --start_t=0.0 --end_t=99.0 --skip_frame=2 --mode=MIN
```
![image-20230909002235029](./img/image-20230909002235029.png)

## 3 MAX Mode
```bash
python composite_image.py --video_path=./example_video.mp4 --start_t=0.0 --end_t=99.0 --skip_frame=2 --mode=MAX
```
![image-20230909002149494](./img/image-20230909002149494.png)

# Extended Version

## composite_image_extended.py

An alternative version with different frame selection approach:

- **Frame Count Control**: Uses `snap_num` parameter to specify the exact number of frames to merge
- **Frame Distribution**: Includes first and last frames in the selection
- **Alpha Blending**: Same transparency control as the original version

### Example with All Parameters
```bash
python composite_image_extended.py --video_path=./example_video.mp4 --start_t=0.0 --end_t=99.0 --snap_num=30 --mode=VAR --alpha=0.7
```

# Parameter

* --video_path: The path of the input video.
* --start_t: the start time of the composite image.
* --end_t: the end time of the composite image.
* --skip_frame: skip frame number when extracting frames, input `1` for not skipping any frame. *(Used in composite_image.py)*
* --snap_num: total number of frames to merge (including first and last frames). *(Used in composite_image_extended.py)*
* --alpha: transparency of moving objects (0.0-1.0). Lower values make moving objects more transparent, blending them more with the background.
* --mode: mode of image merging, three choices is implemented:
  * VAR (Recommend): Use the pixel that is furthest from the mean of the image
  * MAX: Keep the lightest pixel.
  * MIN: Keep the darkest.
