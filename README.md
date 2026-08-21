# 激光雷达植株行间中线循迹

这个目录是一套基于激光雷达的植株行间中线循迹程序，适合小车在两排较为笔直的植株中间行驶。

它包含：

- 从 `/scan` 提取雷达点
- 从左右植株边界拟合出中线
- 通过 `control/fw_mini_controller.py` 直接控制底盘
- 一个用于手动驾驶和自动循迹切换的 GUI
- 一个实时顶视图，可显示雷达点、左右边界和中线

## 文件说明

- `plant_lidar_centerline_follower.py`
  主循迹程序
- `plant_lidar_centerline_gui.py`
  手动控制 + 自动循迹 GUI
- `row_geometry.py`
  共享的雷达行边界拟合与中线计算逻辑

## 运行前提

你当前需要满足：

- 雷达驱动已经启动，并且在发布 `/scan`
- CAN 底盘控制可以正常使用
- `control/fw_mini_controller.py` 已经可以控制这台车

如果 CAN 还没有配置好，可以先执行：

```bash
python3 /home/orangepi/ugv/control/headless_control.py setup-can
```

## 运行主程序

```bash
source /opt/ros/humble/setup.bash
python3 /home/orangepi/ugv/plant_lidar_centerline/plant_lidar_centerline_follower.py
```

## 运行 GUI

```bash
source /opt/ros/humble/setup.bash
python3 /home/orangepi/ugv/plant_lidar_centerline/plant_lidar_centerline_gui.py
```

GUI 当前包含：

- `Connect` / `Disconnect` CAN
- 手动 `Forward`、`Back`、`Left`、`Right`、`STOP`
- 自动循迹参数输入
- 自动前进 / 自动倒车切换
- `Start Auto` / `Stop Auto`
- 底盘反馈与日志
- 实时雷达顶视图

## 建议初始测试命令

```bash
source /opt/ros/humble/setup.bash
python3 /home/orangepi/ugv/plant_lidar_centerline/plant_lidar_centerline_follower.py \
  --row-width 0.8 \
  --speed 0.20 \
  --forward-max 1.8
```

## 参数说明

- `--scan-topic`
  使用的 ROS2 `sensor_msgs/LaserScan` 话题名。
- `--status-topic`
  发布循迹状态 JSON 字符串的话题名。
- `--row-width`
  预计两排植株之间的距离，单位米。这个参数在“只能看到一侧植株”的情况下尤其重要。
- `--min-row-width`
  最小允许行宽。如果左右边界拟合出来的距离比这个还小，就认为结果不可信。
- `--max-row-width`
  最大允许行宽。如果左右边界拟合出来的距离比这个还大，就认为结果不可信。
- `--speed`
  自动循迹模式下的目标前进速度，单位 m/s。
- `--reverse`
  自动模式改为沿着检测到的中线倒车行驶。开启后会发送负速度，并自动调整转向修正方向。
- `--min-speed`
  自动模式因为误差而减速时，允许保留的最小前进速度。
- `--max-wz`
  最大角速度，单位 `deg/s`，用来限制转向过猛。当前默认值为 `1.2`，程序内部会自动转换成 `rad/s` 再做控制计算。
- `--k-lat`
  横向偏差增益。值越大，小车对左右偏离中线的修正越快。
- `--k-heading`
  朝向误差增益。值越大，小车对路线方向不对正的修正越积极。
- `--lookahead-x`
  在前方多远的位置上评估中线偏差。
- `--forward-min`
  参与拟合的最近前向距离，用来忽略太靠近车头的杂点。
- `--forward-max`
  参与拟合的最远前向距离。设小一点可以减少远处杂点影响，设大一点可以看得更远。
- `--lateral-limit`
  参与处理中，车体左右两侧允许纳入计算的最大横向距离。
- `--range-min`
  雷达量测允许的最小距离。
- `--range-max`
  雷达量测允许的最大距离。
- `--bin-size`
  沿前进方向分箱的长度，单位米。算法会在每个箱里提取一个代表性的左右边界点。
- `--min-points`
  在处理窗口中，至少要有多少个扫描点才尝试做行检测。
- `--min-bins`
  某一侧至少要有多少个有效分箱，才会把这一侧拟合成一条线。
- `--center-deadband`
  忽略靠近车辆中心线附近的点，避免把中心杂物误判成植株边界。
- `--left-percentile`
  每个前向分箱中，选取左边界点时使用的百分位数。
- `--right-percentile`
  每个前向分箱中，选取右边界点时使用的百分位数。
- `--slow-error-y`
  当横向误差超过这个值时，自动模式开始减速。
- `--stop-error-y`
  当横向误差超过这个值时，前进速度直接降为 0，优先保证安全。
- `--slow-heading-rad`
  当朝向误差超过这个值时，自动模式开始减速。
- `--control-period`
  控制循环周期，单位秒。
- `--status-period`
  状态发布周期，单位秒。
- `--scan-timeout`
  距离上一帧雷达数据超过这个时间，就认为雷达超时并停车。
- `--lost-hold-s`
  丢失行中线后保持停车的时间。

## 调参建议

如果车左右摆动比较明显：

- 降低 `--k-lat`
- 降低 `--k-heading`
- 降低 `--speed`

如果车回正太慢：

- 提高 `--k-lat`

如果车对路线方向修正不够：

- 提高 `--k-heading`

如果远处杂点太多，影响中线检测：

- 减小 `--forward-max`
- 减小 `--lateral-limit`

## 安全建议

- 丢失中线或雷达超时后，程序会停车
- 第一次实车测试建议把速度设在 `0.15 ~ 0.20 m/s`
- 第一次测试时务必有人在车旁边随时接管
