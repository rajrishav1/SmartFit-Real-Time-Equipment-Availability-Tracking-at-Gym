# 🏋️‍♂️ SmartFit – Real-Time Equipment Availability Tracking at ASU Gym

This projecT focuses on solving the problem of long wait times for gym equipment at ASU by using **real-time pose estimation** to monitor equipment usage.

## 🔍 Problem Overview

- Students experience long waits for popular gym equipment like treadmills and cycles.  
- No current system provides real-time updates on equipment status, leading to inefficient workouts and overcrowding.  
- There is a need for an automated solution to reduce congestion and improve gym user experience.

## 🛠 Proposed Solution

- Install cameras focused on gym equipment zones.  
- Use **YOLOv8n-Pose**, a lightweight and fast pose estimation model, to detect human foot positions in real time.  
- Check whether feet are inside predefined gym equipment zones using geometric analysis (`Shapely`).  
- Display real-time occupancy status (future enhancement: mobile/web dashboard).

## ⚙️ System Components

- **1_select_and_save_zones.py**  
  - Allows users to manually annotate gym zones on video frames and save zone definitions to a JSON file.

- **2_run_inference_with_zones.py**  
  - Runs real-time inference on gym videos using YOLOv8n-Pose.  
  - Checks occupancy of each zone by detecting foot keypoints.  
  - Displays visual overlays showing occupied/unoccupied status for each piece of equipment.

## 📈 Results

- Successful desktop proof-of-concept tested on gym video files.  
- Accurate detection of equipment usage, even in busy scenes.  
- Visual validation confirmed correct alignment of pose detection with real-world usage.

## 💡 Limitations and Mitigation

- **Limitations**: Occlusions between users, lighting changes, no facial detection (privacy compliant).  
- **Mitigation**: Adjust camera angles, retrain models periodically, ensure secure data handling for live feeds.

## 🚀 Future Enhancements

- Integrate live camera feeds for real-time monitoring.  
- Develop a mobile or web-based dashboard for users to check gym status.  
- Add historical analytics and crowd forecasting features.  
- Improve model robustness in crowded or low-light conditions.

This project combines computer vision, pose estimation, and real-time system design to improve the gym experience for ASU students.  
