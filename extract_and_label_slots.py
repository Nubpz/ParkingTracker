"""
extract_and_label_slots.py
Automatically crop and label parking slots from a video
based on known empty slot indices.
This version clears old images before regenerating a new dataset.
"""

import cv2
import pickle
import os

# Configuration
video_path = "parking_lot_video.mp4"     # Input video
pkl_path = "data/slot_positions.pkl"            # Slot coordinates
output_root = "train_data/train"                # Output base folder

# Slot dimensions (must match those in annotation)
slot_w, slot_h = 120, 45

# Process every Nth frame (to avoid redundant frames)
frame_interval = 30  

# Define empty slots from your reference frame
empty_slots = [6, 26, 33, 39, 42, 43, 52, 54, 56, 58, 65, 66]

# Prepare directories
empty_dir = os.path.join(output_root, "empty")
occupied_dir = os.path.join(output_root, "occupied")
os.makedirs(empty_dir, exist_ok=True)
os.makedirs(occupied_dir, exist_ok=True)

# Clear any existing images before regenerating
for folder in [empty_dir, occupied_dir]:
    for f in os.listdir(folder):
        file_path = os.path.join(folder, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
print("Old dataset cleared from 'empty' and 'occupied' folders.\n")

# Load slot positions
with open(pkl_path, "rb") as f:
    slot_positions = pickle.load(f)

if not slot_positions:
    raise Exception("No slot positions found in slot_positions.pkl.")

print(f"Total slots: {len(slot_positions)}")
print(f"Empty slots: {empty_slots}\n")

# Process video frames and crop slots
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    if frame_count % frame_interval == 0:
        frame = cv2.resize(frame, (1280, 720))

        for i, (x, y) in enumerate(slot_positions):
            crop = frame[y:y + slot_h, x:x + slot_w]
            if crop.size == 0:
                continue

            # Choose target folder based on slot status
            folder = empty_dir if i in empty_slots else occupied_dir

            # Compact filename format: s<slot#>_f<frame#>.jpg
            filename = f"s{i:02d}_f{frame_count//frame_interval:04d}.jpg"
            cv2.imwrite(os.path.join(folder, filename), crop)
            saved_count += 1

    frame_count += 1

cap.release()
print(f"Video processing complete.")
print(f"Total frames read: {frame_count}")
print(f"Total cropped slot images saved: {saved_count}")
print(f"\nSaved datasets:")
print(f"  {empty_dir}")
print(f"  {occupied_dir}")
