import cv2
import numpy as np
import json

# ========== USER SETTINGS ==========
video_path = "/home/pranaypalem/Downloads/Gym_YoloPose3.mp4"
save_path = "zones.json"
# ====================================

zones = []
current_polygon = []
current_label = 'treadmill'

def draw_polygon(event, x, y, flags, param):
    global current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and current_polygon:
        zones.append((current_polygon[:], current_label))
        current_polygon = []

# ========== Open Video and Select Zones ==========
cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
if not ret:
    raise Exception("Failed to read video")

cv2.namedWindow("Select Zones")
cv2.setMouseCallback("Select Zones", draw_polygon)

while True:
    temp = first_frame.copy()
    if current_polygon:
        cv2.polylines(temp, [np.array(current_polygon, np.int32)], False, (255, 0, 0), 2)
    for poly, label in zones:
        cv2.polylines(temp, [np.array(poly, np.int32)], True, (0, 255, 255), 2)
        cv2.putText(temp, label, poly[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Select Zones", temp)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('t'):
        current_label = 'treadmill'
    elif key == ord('c'):
        current_label = 'cycle'

cv2.destroyWindow("Select Zones")
cap.release()

# ========== Save zones to file ==========
save_data = [{"polygon": poly, "label": label} for poly, label in zones]

with open(save_path, 'w') as f:
    json.dump(save_data, f)

print(f"Zones saved to {save_path}")
