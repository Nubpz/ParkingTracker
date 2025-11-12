"""
annotate_slots.py
Step 1: Manually define parking slot regions (ROIs).

Instructions:
- Left mouse click  : Add a new slot
- Right mouse click : Remove an existing slot
- Press 'q'         : Save and quit
"""

import cv2
import pickle
import os

# Configuration
# Path to a reference video or image frame
frame_path = "videos/parking_lot_video.mp4"

# Slot dimensions (in pixels)
slot_w, slot_h = 120, 45

# Directory to save cropped slot images
save_dir = "data/cropped_slots"
os.makedirs(save_dir, exist_ok=True)

# Load a reference frame from the video
cap = cv2.VideoCapture(frame_path)
success, frame = cap.read()
cap.release()

if not success:
    raise Exception("Could not read the video frame. Check the file path.")

# Resize for convenience (optional, adjust as needed)
frame = cv2.resize(frame, (1280, 720))

# Load existing slot positions if available
pkl_path = "data/slot_positions.pkl"
if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as f:
        slot_positions = pickle.load(f)
else:
    slot_positions = []

# Mouse callback for adding/removing parking slots
def mouse_click(event, x, y, flags, param):
    """Handles mouse events for adding or removing slot positions."""
    global slot_positions

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add a new slot at the clicked location
        slot_positions.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove slot if the click is inside an existing box
        for i, pos in enumerate(slot_positions):
            x1, y1 = pos
            if x1 < x < x1 + slot_w and y1 < y < y1 + slot_h:
                slot_positions.pop(i)
                break

    # Save updated slot list
    with open(pkl_path, "wb") as f:
        pickle.dump(slot_positions, f)

# Annotation loop
cv2.namedWindow("Annotate Parking Slots")
cv2.setMouseCallback("Annotate Parking Slots", mouse_click)

while True:
    temp = frame.copy()

    # Draw current slot rectangles
    for pos in slot_positions:
        cv2.rectangle(temp, pos, (pos[0] + slot_w, pos[1] + slot_h), (255, 0, 255), 2)

    cv2.imshow("Annotate Parking Slots", temp)

    key = cv2.waitKey(1)
    if key == ord("q"):
        # Quit the loop and save results
        break

cv2.destroyAllWindows()

# Save cropped slot images
for i, (x, y) in enumerate(slot_positions):
    roi = frame[y:y + slot_h, x:x + slot_w]
    cv2.imwrite(f"{save_dir}/slot_{i}.png", roi)

print(f"Saved {len(slot_positions)} slot coordinates to {pkl_path}")
print(f"Cropped images saved in {save_dir}/")
