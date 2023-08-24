from ultralytics import YOLO
import supervision as sv
import cv2
import cvzone
import torch
from ultralytics.yolo.utils.plotting import Annotator
import time
import imutils

VIDEOPATH = 'crowd_accident.mp4'
CCTV_RTSP = ''
TIMER = 5 #Seconds
PERSON_LIMIT = 5

crowd_detector = YOLO("yolov8l.pt")
# crowd_detector.fuse()

######## VIDEO CAPTURE
video = cv2.VideoCapture(VIDEOPATH)

fpsReader = cvzone.FPS()

print(torch.cuda.is_available())

box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

class_list = crowd_detector.model.names
classes = ['person']
prev = time.time()

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_save = cv2.VideoWriter('output/output.mp4', fourcc, 30, (1680, 945))

while True:

    #Read a new frame
    ret, frame = video.read()

    # Check if frame is read successfully
    if not ret:
        continue

    ### Window Resize
    height, width = frame.shape[:2]
    cv2.namedWindow('Crowd Gathering', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Crowd Gathering', 1680, 945)

    ### FPS update
    fps, frame = fpsReader.update(frame, pos=(15,30), color = (0,255,0), scale = 2, thickness= 2)
    
    ### Detections 
    annotator = Annotator(frame)
    crowd_detections = crowd_detector.predict(frame, device = 0, tracker='bytetrack.yaml')
    personCount = {}
    timer = 0

    for result in crowd_detections:
        boxes = result.boxes.cpu().numpy()
        
        for box in boxes:
            (x, y, w ,h) = box.xyxy[0]

            ### Finding Centre Points
            cx = int((x + w )/ 2)
            cy = int((y + h) / 2)
            cy_bot = int(h)
            # cv2.circle(frame, (cx,cy), 3, (0,255,0), -1)

            ### Class Declaration
            b = box.xyxy[0].astype(int)
            c = int(box.cls[0])
            class_names = class_list[int(c)]
            # id = int(box.id[0]) if box.id is not None else 0 

            if class_names == 'person':
                if not c in personCount.keys():
                    personCount[c] = 1
                else:
                    personCount[c] += 1

                annotator.box_label(b, color=(255,0,0))
            
        for key in personCount.keys():
            # print(class_list[key] + " : " + str(personCount[key]))
            cvzone.putTextRect(frame, 'Person: ' + str(personCount[key]), (200,20), 1, thickness=2, colorT= (255,255,255), colorR= (0,0,0), font= cv2.FONT_HERSHEY_PLAIN  )
            # cv2.putText(frame, 'Person Count: ' + str(personCount[key]), (150,30), cv2.FONT_HERSHEY_COMPLEX, 1, color = (255,255,255), thickness= 1)
       
            if personCount[key] > PERSON_LIMIT:
                curr =  time.time()
                if curr-prev > TIMER:
                    elapsed_time = curr-prev
                    text_area = (1350, 80)
                    cvzone.putTextRect(frame, 'Crowd Detected', text_area, 2, thickness=3, colorT=(255,255,255), colorR = (0,0,255), font=cv2.FONT_HERSHEY_DUPLEX)
            else:
                prev = curr
    
    # video_save.write(frame)
    cv2.imshow('Crowd Gathering', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
# video.save_release()
cv2.destroyAllWindows()
    