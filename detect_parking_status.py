"""
detect_parking_status.py
- shows every frame
- only predicts every N frames
- draws timers every frame
"""
#this file uses larger model model "parking_model.h5"
#run as "python3 detect_parking_status.py" this modal is slowest one ....Try changing the line line 25 to "models/parking_model_SMALL.h5"
import cv2
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model


# convert seconds to HH:MM:SS
def fmt(sec:float):
    h=int(sec//3600)
    m=int((sec%3600)//60)
    s=int(sec%60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# config
model_path = "models/parking_model.h5"
video_path = "videos/parking_lot_video.mp4"
pkl_path   = "data/slot_positions.pkl"

slot_w,slot_h = 120,45
img_w,img_h   = 96,96
resize_width,resize_height = 1280,720

predict_every = 30      # predict only every 30 frames


print("loading model / slots ...")
model = load_model(model_path)
model.build((None,48,48,3))  # helps keras shape
predict_fn = model.__call__

with open(pkl_path,"rb") as f:
    slot_positions = pickle.load(f)

print("q = quit")


# timer memory for each slot
slot_timers={i:{"start":None,"duration":0.0} for i in range(len(slot_positions))}

# last labels cache
last_labels=[0]*len(slot_positions)


cap=cv2.VideoCapture(video_path)
fps=int(cap.get(cv2.CAP_PROP_FPS)) or 30
delay=int(1000/fps)
frame_idx=0
t0=time.time()


while True:
    ret,frame=cap.read()
    if not ret: break
    frame_idx+=1

    frame=cv2.resize(frame,(resize_width,resize_height))

    now=time.time()

    # ------------------------------------
    # only crop + predict every N frames
    # ------------------------------------
    if frame_idx % predict_every == 0:
        crops=[]
        for (x,y) in slot_positions:
            roi=frame[y:y+slot_h, x:x+slot_w]
            if roi.size==0: continue
            # do resize only here (not every frame)
            crop=cv2.resize(roi,(img_w,img_h))/255.0
            crops.append(crop)

        if crops:
            preds=predict_fn(np.array(crops,dtype=np.float32),training=False).numpy()
            last_labels=np.argmax(preds,axis=1).tolist()

    # use last_labels every frame
    free_cnt=0
    occ_cnt=0

    for i,lab in enumerate(last_labels):
        (x,y)=slot_positions[i]
        if lab==1:
            occ_cnt+=1
            t=slot_timers[i]
            if t["start"] is None: t["start"]=now
            t["duration"]=now-t["start"]
        else:
            free_cnt+=1
            slot_timers[i]["start"]=None
            slot_timers[i]["duration"]=0.0

        # draw box
        # color=(0,255,0) if lab==0 else (0,0,255)
        # cv2.rectangle(frame,(x,y),(x+slot_w,y+slot_h),color,2)
        color=(0,255,0) if lab==0 else (0,0,255)
        cx=x+slot_w//2
        cy=y+slot_h//2
        cv2.circle(frame,(cx,cy),9,color,-1) 

        # draw timer text if occupied
        # if lab==1:
        #     txt=fmt(slot_timers[i]["duration"])
        #     (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        #     tx=x+(slot_w-tw)//2
        #     ty=y+(slot_h+th)//2
        #     cv2.putText(frame,txt,(tx,ty),
        #                 cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        if lab==1:
            txt=fmt(slot_timers[i]["duration"])
            (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.6,1)
            cv2.putText(frame,txt,(cx-tw//2,cy+18),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)


    # draw counters
    cv2.rectangle(frame,(0,0),(260,70),(255,255,255),-1)
    cv2.putText(frame,f"FREE: {free_cnt}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,200,0),2)
    cv2.putText(frame,f"OCCUPIED: {occ_cnt}",(10,60),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)


    cv2.imshow("Parking Tester SMOOTH",frame)
    if cv2.waitKey(delay)&0xFF==ord('q'): break

cap.release()
cv2.destroyAllWindows()


# """
# detect_parking_status.py
# Optimized + slot parking duration timer
# Compatible with 48x48 model
# """

# import cv2
# import numpy as np
# import pickle
# import time
# from tensorflow.keras.models import load_model


# # FORMAT TIME
# def format_duration(seconds):
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     sec = int(seconds % 60)
#     return f"{hours:02d}:{minutes:02d}:{sec:02d}"


# # Configuration
# model_path = "models/parking_model_SMALL.h5"
# video_path = "videos/parking_lot_video.mp4"
# pkl_path   = "data/slot_positions.pkl"

# slot_w, slot_h = 120, 45
# img_w, img_h = 96, 96
# frame_skip = 30
# resize_width, resize_height = 1280, 720

# print("Loading model and slots...")
# model = load_model(model_path)
# predict_fn = model.__call__
# with open(pkl_path,"rb") as f:
#     slot_positions = pickle.load(f)

# print(f"Loaded {len(slot_positions)} slots\n")

# # TIMER MEMORY FOR EACH SLOT
# slot_timers = {i: {"start":None, "duration":0.0} for i in range(len(slot_positions))}

# # Video
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     raise IOError("Cannot open video file.")

# fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
# delay = int(1000/fps)
# cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
# print("Press 'q' to quit.")

# frame_idx = 0
# t_start = time.time()

# while True:
#     ret, frame = cap.read()
#     if not ret: break
#     frame_idx+=1
#     if frame_idx % frame_skip != 0: continue

#     # resize + normalize frame
#     frame = cv2.resize(frame,(resize_width,resize_height))
#     f32 = frame.astype("float32")/255.0

#     crops, valid_pos = [],[]
#     for (x,y) in slot_positions:
#         roi = f32[y:y+slot_h, x:x+slot_w]
#         if roi.size==0: continue
#         crop = cv2.resize(roi,(img_w,img_h))
#         crops.append(crop)
#         valid_pos.append((x,y))

#     if crops:
#         preds = predict_fn(np.array(crops,dtype="float32"),training=False).numpy()
#         labels = np.argmax(preds,axis=1)
#     else:
#         labels=[]

#     # UPDATE TIMERS
#     now = time.time()
#     for i,label in enumerate(labels):
#         t = slot_timers[i]
#         if label==1: # OCCUPIED
#             if t["start"] is None: t["start"] = now
#             t["duration"] = now - t["start"]
#         else:        # FREE
#             t["start"] = None
#             t["duration"] = 0.0

#     # counters
#     free_count     = int(np.sum(labels==0))
#     occupied_count = int(np.sum(labels==1))

#     # draw
#     for idx,((x,y),label) in enumerate(zip(valid_pos,labels)):
#         color = (0,255,0) if label==0 else (0,0,255)
#         cv2.rectangle(frame,(x,y),(x+slot_w,y+slot_h),color,2)

#         if label==1:
#             dur = slot_timers[idx]["duration"]
#             time_str = format_duration(dur)

#             font_scale = 0.5 
#             font_thickness = 1
#             text_color = (255, 255, 255)

#             (text_w, text_h), baseline = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#             text_x = x + (slot_w - text_w) // 2
#             text_y = y + (slot_h + text_h) // 2

#             cv2.putText(frame, time_str, (text_x, text_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

#     # COUNTER DISPLAY LOGIC
#     cv2.rectangle(frame,(0,0),(260,70),(255,255,255),-1)
#     cv2.putText(frame,f"FREE: {free_count}",(10,30),
#                 cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,200,0),2)
#     cv2.putText(frame,f"OCCUPIED: {occupied_count}",(10,60),
#                 cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

#     cv2.imshow("Parking Lot Status (Fast)",frame)
#     if cv2.waitKey(delay)&0xFF==ord('q'): break

# cap.release()
# cv2.destroyAllWindows()

# t_total = time.time() - t_start
# print(f"\nProcessed {frame_idx} frames in {t_total:.2f}s "
#       f"({frame_idx/t_total:.2f} FPS avg)")
