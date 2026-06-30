# leidaautorun

This repository contains the related `autorunlida` integration files for:

- lidar local row-following
- ROS local command publishing
- hybrid autorun backend integration

Current contents:

- `autorunlida/backend.py`
- `autorunlida/plant_lidar_centerline_ros_follower.py`
- `autorunlida/row_geometry.py`
- `autorunlida/gui_settings.json`

Notes:

- `plant_lidar_centerline_ros_follower.py` is the ROS-output local follower variant used by `autorunlida`
- `backend.py` contains the hybrid autorun logic and local/global blending path
- this upload is a minimal related-code snapshot, not the full runtime workspace
