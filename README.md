# autorunlida

`autorunlida` 把定位、录制、混合驱动收成一个可直接使用的工程：

- Odin / 全局定位与任务执行
- 激光雷达 / 行间局部中线约束

## 设计原则

- 行间正常前进时：
  - 使用激光雷达局部中线控制
- 起步、行尾、倒车过渡和横移换行时：
  - 使用分阶段的全局路径控制
- 横移完成准备入行时：
  - 停车确认双侧边界，再由激光低速入行

## 目录

- 任务与日志写入 `autorunlida/`
- Mission 录制、定位、混合驱动都直接由 `autorunlida` 提供

## 常用命令

### 建图

```bash
cd /home/orangepi/ugv/autorunlida
python3 main.py map --map-name lab_map
```

### 纯定位

```bash
python3 main.py localization \
  --db /home/orangepi/ugv/autorunlida/maps/lab_map/lab_map.bin
```

### 录制全局任务

```bash
python3 main.py record \
  --db /home/orangepi/ugv/autorunlida/maps/lab_map/lab_map.bin \
  --mission-name mission_a
```

### 混合驱动

```bash
python3 main.py autorun \
  --db /home/orangepi/ugv/autorunlida/maps/lab_map/lab_map.bin \
  --mission /home/orangepi/ugv/autorunlida/missions/mission_a.json
```

## 关键参数

- `--line-cruise-vx`: 正常行内巡航速度
- `--lidar-row-entry-*`: 换行后的停车识别和低速入行参数
- `--lidar-*`: 激光雷达局部中线跟踪参数

## 当前实现说明

- 局部控制由进程内的 `plant_lidar_centerline_follower.py` 提供
- 正常行内使用激光控制；任务过渡阶段使用全局路径控制
- 最终底盘 CAN 命令由 `autorunlida` 统一下发
