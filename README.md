# Crowd Gathering using YOLOv8

## Feature
- Detects the crowd gathered in the cctv

## Usage 
```
pip install -r requirements.txt
python3 crowd_detection.py
```

## To use other videos / rtsp
- On line 9 and 10, change the video/rtsp path you desired
```
CCTV_RTSP = "rtsp://XXX.XXX.XXX.XXX/channel/XXX'
VIDEO_PATH = "crowd_accident.mp4"
```
- Line 18, choose either want RTSP (CCTV_RSTP) or Video (VIDEOPATH)

## Change the timer and total person to make it detect as the crowd gathering
- Change the values at line 11 and line 12 you desired

## Methodology
- This module uses [YOLOv8](https://docs.ultralytics.com/) as pretrained detection model with ImageNet. The scripts will detect the total person in the video/rtsp, if it reach the total limit of the person in the video/rtsp within few seconds, it will triggered as crowd detected.

