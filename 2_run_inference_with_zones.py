import cv2
import numpy as np
import json
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

# ========== USER SETTINGS ==========
video_path = "/home/pranaypalem/Downloads/Gym_YoloPose3.mp4"
pose_model_path = "yolov8n-pose.pt"
zone_file = "zones.json"
transparency = 0.4
# ====================================

# ========== Load Zones ==========
with open(zone_file, 'r') as f:
    saved_zones = json.load(f)

zones = [(zone["polygon"], zone["label"]) for zone in saved_zones]
zone_labels = [label for _, label in zones]
total_counts = {label: zone_labels.count(label) for label in set(zone_labels)}
occupied_counts = {label: 0 for label in total_counts}

def draw_zones(img, zone_status):
    overlay = img.copy()
    for i, (poly, label) in enumerate(zones):
        color = (0, 255, 0) if not zone_status[i] else (0, 0, 255)
        poly_np = np.array(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [poly_np], color)

        M = cv2.moments(poly_np)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(overlay, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return cv2.addWeighted(overlay, transparency, img, 1 - transparency, 0)

# ========== Load model and video ==========
model = YOLO(pose_model_path)
cap = cv2.VideoCapture(video_path)

cv2.namedWindow("Original | Inference", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    keypoints = results.keypoints.xy if results.keypoints else []
    zone_status = [False] * len(zones)

    if keypoints is not None:
        for kp in keypoints:
            if len(kp) > 15:
                feet = [kp[15].tolist(), kp[16].tolist()]
                for i, (poly, _) in enumerate(zones):
                    polygon = Polygon(poly)
                    for foot in feet:
                        if polygon.contains(Point(foot)):
                            zone_status[i] = True
                            break

    occupied_counts = {label: 0 for label in total_counts}
    for (poly, label), status in zip(zones, zone_status):
        if status:
            occupied_counts[label] += 1

    frame_with_pose = results.plot()
    frame_with_zones = draw_zones(frame_with_pose, zone_status)

    y0 = 30
    for i, label in enumerate(total_counts):
        text = f"{label}: {occupied_counts[label]}/{total_counts[label]}"
        y = y0 + i * 35
        cv2.rectangle(frame_with_zones, (frame.shape[1]-250, y-25), (frame.shape[1]-10, y+5), (255, 255, 255), -1)
        cv2.putText(frame_with_zones, text, (frame.shape[1]-240, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    if frame.shape != frame_with_zones.shape:
        frame = cv2.resize(frame, (frame_with_zones.shape[1], frame_with_zones.shape[0]))
    combined = np.concatenate((frame, frame_with_zones), axis=1)

    cv2.imshow("Original | Inference", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
