# autorunlida

`autorunlida` 把定位、录制、混合驱动收成一个可直接使用的工程：

- Odin / 全局定位与任务执行
- 激光雷达 / 行间局部中线约束

## 设计原则

- 行间正常前进时：
  - 以激光雷达局部跟踪为主
  - 同时保留一部分全局定位回放权重
- 换行、横移、倒车、到路尽头、找不到中线时：
  - 切到全局定位主导
  - 局部权重降为 0

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

- `--local-weight-in-row`: 行间局部控制权重，默认 `0.75`
- `--global-weight-in-row`: 行间全局控制权重，默认 `0.25`
- `--lidar-*`: 直接传给激光雷达局部跟踪的主要参数

## 当前实现说明

- 局部控制通过 `lidar_local_runner.py` 运行
- `autorunlida` 订阅局部跟踪的状态和 `cmd_vel`
- 最终底盘 CAN 命令由 `autorunlida` 统一下发
- 行间前进段优先采用局部输出；换行/倒车/crab 段只走全局
