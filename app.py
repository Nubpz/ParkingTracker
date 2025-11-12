"""
app.py – Selective Inference + class mapping + confidence gate
smooth video, timers+boxes only
"""

import os, time, threading
from datetime import datetime
import cv2
import numpy as np
import pickle
from flask import Flask, render_template, Response, jsonify, url_for
from tensorflow.keras.models import load_model

# ----------------- CONFIG -----------------
app = Flask(__name__, template_folder='client', static_folder='static')

LOTS = {
    "Lot1": "videos/parking_lot_video.mp4",
    "Lot2": "videos/parking_lot_videoR.MOV",
    "Lot3": "videos/parking_lot_video.mp4"
}

MODEL_PATH   = "models/parking_model_SMALL.h5"  # softmax (2 classes)
SLOTS_PKL    = "data/slot_positions.pkl"
RESIZE       = (1280, 720)                      # must match annotation size
SLOT_SIZE    = (120, 45)
IMG_SIZE     = (96, 96)

# Selective inference controls
FRAME_SKIP_INFER   = 15      # run model every N frames (smoothness)
FULL_REFRESH_EVERY = 120     # force full re-eval of all slots every N frames
DIFF_THRESH        = 10.0    # brightness diff threshold (lower = more updates)

# Softmax class mapping / stabilization
OCCUPIED_CLASS_INDEX = 1     # <<< if inverted, set to 0
CONFIDENCE_MIN       = 0.60  # if max prob < threshold, keep previous label

# Streaming
JPEG_QUALITY   = 65
STREAM_SLEEP_S = 0.015       # ~66 FPS feel

latest_jpeg   = {}
parking_stats = {}
state_lock    = threading.Lock()

# ----------------- UTILS -----------------
def load_slots():
    with open(SLOTS_PKL, "rb") as f:
        return pickle.load(f)

def format_duration(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

def mean_brightness(img):
    # accept empty roi defensively
    if img.size == 0: return 0.0
    return float(np.mean(img))

# WORKER 
class LotWorker(threading.Thread):
    def __init__(self, lot_name, video_path):
        super().__init__(daemon=True)
        self.lot_name  = lot_name
        self.video_path= video_path
        self.stop_flag = threading.Event()

    def run(self):
        print(f"[INFO] starting worker for {self.lot_name}")
        model   = load_model(MODEL_PATH)
        slots   = load_slots()
        timers  = {i: {"start": None, "dur": 0.0} for i in range(len(slots))}
        prev_fr = None

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video for {self.lot_name}")
            return

        predict_fn  = model.__call__
        frame_idx   = 0
        slot_w, slot_h = SLOT_SIZE
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

        # cached labels & probs (start as free)
        cached_label = [0] * len(slots)
        cached_prob  = [1.0] * len(slots)  # confidence of current label

        while not self.stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, RESIZE)
            frame_idx += 1

            # compute per-slot brightness diffs (vs previous displayed frame)
            diffs = []
            if prev_fr is not None:
                for (x, y) in slots:
                    crop_now  = frame[y:y+slot_h, x:x+slot_w]
                    crop_prev = prev_fr[y:y+slot_h, x:x+slot_w]
                    diffs.append(abs(mean_brightness(crop_now) - mean_brightness(crop_prev)))
            else:
                # first pass: force inference for all
                diffs = [1e9] * len(slots)

            # decide which slots to update this cycle
            need_idxs = []
            do_full_refresh = (frame_idx % FULL_REFRESH_EVERY == 0)

            if frame_idx % FRAME_SKIP_INFER == 0 or do_full_refresh:
                if do_full_refresh:
                    need_idxs = list(range(len(slots)))
                else:
                    # changed slots only
                    for i, dv in enumerate(diffs):
                        if dv > DIFF_THRESH:
                            need_idxs.append(i)

                if need_idxs:
                    crops = []
                    for i in need_idxs:
                        x, y = slots[i]
                        roi = frame[y:y+slot_h, x:x+slot_w]
                        roi = cv2.resize(roi, IMG_SIZE) / 255.0
                        crops.append(roi)

                    if crops:
                        preds = predict_fn(np.array(crops, dtype="float32"), training=False).numpy()
                        # softmax → probs per class
                        # derive class + max prob, then map with OCCUPIED_CLASS_INDEX
                        for k, i in enumerate(need_idxs):
                            prob_vec = preds[k]
                            cls = int(np.argmax(prob_vec))
                            conf = float(np.max(prob_vec))

                            # confidence gate: if unsure, keep existing label
                            if conf < CONFIDENCE_MIN:
                                # keep cached
                                continue

                            # store new label, but convert to occupied/free via mapping
                            # occupied if cls == OCCUPIED_CLASS_INDEX
                            new_lab = 1 if cls == OCCUPIED_CLASS_INDEX else 0
                            cached_label[i] = new_lab
                            cached_prob[i]  = conf

                # only update prev_fr when we do an inference cycle
                prev_fr = frame.copy()

            # draw & timers
            free = 0
            occ  = 0
            now = time.time()

            for i, (x, y) in enumerate(slots):
                lab = cached_label[i]  # 0=free, 1=occupied after mapping

                color = (0, 255, 0) if lab == 0 else (0, 0, 255)
                #cv2.rectangle(frame, (x, y), (x + slot_w, y + slot_h), color, 2)
                cx = x + slot_w//2
                cy = y + slot_h//2
                cv2.circle(frame, (cx, cy), 9, color, -1)

                if lab == 1:
                    occ += 1
                    t = timers[i]
                    if t["start"] is None:
                        t["start"] = now
                    t["dur"] = now - t["start"]
                    txt = format_duration(t["dur"])
                    # (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # tx = x + (slot_w - tw) // 2
                    # ty = y + (slot_h + th) // 2
                    # cv2.putText(frame, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # draw timer under circle
                    txt = format_duration(t["dur"])
                    (cv_w, cv_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.putText(frame, txt, (cx - cv_w//2, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                else:
                    free += 1
                    timers[i]["start"] = None
                    timers[i]["dur"] = 0.0

            ok, jpg = cv2.imencode(".jpg", frame, encode_param)
            if ok:
                with state_lock:
                    latest_jpeg[self.lot_name] = jpg.tobytes()
                    parking_stats[self.lot_name] = {
                        "total": len(slots),
                        "free": int(free),
                        "occupied": int(occ),
                        "updated_at": datetime.utcnow().isoformat() + "Z",
                    }

        cap.release()
        print(f"[INFO] worker stopped for {self.lot_name}")

# ROUTES 
@app.route("/")
def home():
    return render_template(
        "index.html",
        lots=LOTS.keys(),
        lot_links={lot: url_for("lot_view", lot_name=lot) for lot in LOTS.keys()},
    )

@app.route("/lot/<lot_name>")
def lot_view(lot_name: str):
    if lot_name not in LOTS:
        return "Unknown lot", 404
    return render_template(
        "lot.html",
        lot=lot_name,
        stream_url=url_for("video_stream", lot_name=lot_name),
    )

@app.route("/stream/<lot_name>")
def video_stream(lot_name: str):
    if lot_name not in LOTS:
        return "Unknown lot", 404

    def gen():
        boundary = b"--frame\r\n"
        while True:
            with state_lock:
                frame_bytes = latest_jpeg.get(lot_name, b"")
            if frame_bytes:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            time.sleep(STREAM_SLEEP_S)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api_stats")
def api_stats():
    with state_lock:
        snapshot = {lot: dict(v) for lot, v in parking_stats.items()}
    return jsonify(snapshot)

# MAIN 
if __name__ == "__main__":
    os.makedirs("client", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    # initialize placeholders so UI doesn't error before first frame
    try:
        total = len(load_slots())
    except Exception:
        total = 0
    with state_lock:
        for lot in LOTS:
            latest_jpeg[lot] = b""
            parking_stats[lot] = {"total": total, "free": total, "occupied": 0, "updated_at": datetime.utcnow().isoformat() + "Z"}

    for lot, vid in LOTS.items():
        t = LotWorker(lot, vid)
        t.start()
        print(f"[INFO] worker thread started for {lot}")

    app.run(debug=True, threaded=True)