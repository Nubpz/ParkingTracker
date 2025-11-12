#this file uses smaller model "parking_model_INT8.tflite"
#run as "python3 detect_parking_status_small.py"
import cv2
import numpy as np
import pickle
import time
import tensorflow as tf

# CONFIG
tfl_path = "models/parking_model_INT8.tflite"
video_path = "videos/parking_large.mp4"
pkl_path   = "data/slot_positions.pkl"

slot_w, slot_h = 120, 45
img_w, img_h   = 96, 96
frame_skip     = 10
resize_width, resize_height = 1280, 720

# TIMER FORMAT
def format_duration(sec):
    h = int(sec//3600); m=int((sec%3600)//60); s=int(sec%60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# LOAD MODEL
interpreter = tf.lite.Interpreter(model_path=tfl_path)
interpreter.allocate_tensors()
input_idx  = interpreter.get_input_details()[0]['index']
output_idx = interpreter.get_output_details()[0]['index']

def tfl_predict(batch_f32):
    # batch_f32 shape: (N,96,96,3) float32 0-1
    # quantize to int8
    input_details = interpreter.get_input_details()[0]
    scale, zero_point = input_details["quantization"]
    batch_int8 = batch_f32/scale + zero_point
    batch_int8 = batch_int8.astype(np.int8)

    interpreter.resize_tensor_input(input_idx, batch_int8.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_idx, batch_int8)
    interpreter.invoke()
    out = interpreter.get_tensor(output_idx)
    return out

# LOAD SLOTS
with open(pkl_path,'rb') as f: slot_positions = pickle.load(f)
slot_timers = {i: {"start":None, "duration":0.0} for i in range(len(slot_positions))}

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
delay=int(1000/fps)
frame_idx=0; t0=time.time()

print("Press q to quit…")

while True:
    ret,frame=cap.read()
    if not ret: break
    frame_idx+=1
    if frame_idx % frame_skip !=0: continue

    frame=cv2.resize(frame,(resize_width,resize_height))

    crops=[]; pos=[]
    for (x,y) in slot_positions:
        roi=frame[y:y+slot_h, x:x+slot_w]
        if roi.size==0: continue
        crops.append(cv2.resize(roi,(img_w,img_h))/255.0)
        pos.append((x,y))

    if crops:
        preds = tfl_predict(np.array(crops, dtype=np.float32))
        labels = np.argmax(preds,axis=1)
    else:
        labels=[]

    now=time.time()
    free=np.sum(labels==0)
    occ=np.sum(labels==1)

    # update timers + draw
    for i,((x,y),lab) in enumerate(zip(pos,labels)):
        t=slot_timers[i]
        if lab==1:
            if t["start"] is None: t["start"]=now
            t["duration"]=now-t["start"]
        else:
            t["start"]=None; t["duration"]=0.0

        color=(0,255,0) if lab==0 else (0,0,255)
        cv2.rectangle(frame,(x,y),(x+slot_w,y+slot_h),color,2)
        if lab==1:
            txt=format_duration(t["duration"])
            (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            tx=x+(slot_w-tw)//2; ty=y+(slot_h+th)//2
            cv2.putText(frame,txt,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    cv2.rectangle(frame,(0,0),(260,70),(255,255,255),-1)
    cv2.putText(frame,f"FREE: {free}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,200,0),2)
    cv2.putText(frame,f"OCCUPIED: {occ}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

    cv2.imshow("INT8 parking",frame)
    if cv2.waitKey(delay)&0xFF==ord('q'): break

cap.release(); cv2.destroyAllWindows()
print("done")
