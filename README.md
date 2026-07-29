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

### 网页手柄控制

```bash
cd /home/orangepi/ugv/autorunlida
./run_web.sh
```

用电脑或手机打开网页，把 Xbox/XInput 手柄连接到打开网页的设备。浏览器检测到手柄后会自动取得控制，无需点击启用按钮；首次连接后按一下手柄按键让浏览器识别设备。

- `A`: 四轮转向
- `B`: 横移
- `X`: 驻车
- `Y`: 空挡
- `RT`: 持续按住才允许运动
- 右摇杆上下：前进、后退
- 四轮转向模式下，左摇杆左右控制转向
- 横移模式下，左摇杆左右只连续控制轮胎角度，满量程对应左右 `90°`、回中即 `0°`；右摇杆只控制前进/后退，两根摇杆互不锁存

网页手柄超过 0.45 秒没有新指令会自动发送零速停车。实物遥控器、急停和自动驾驶拥有更高优先级；启动混合驱动时会自动释放网页手柄控制。

网页服务启动后会持续打开中间的 UVC 相机预览，不需要先启动自动驾驶；停止任务不会关闭预览，相机发布进程意外退出时会自动重试。

网页服务启动时还会按照 `gui_settings.json` 中的 `can_channel` 和 `can_bitrate` 自动连接CAN；如果接口已经启动则直接复用，不会先关闭再重连。默认使用 `can0`、`500000` bps。

## 关键参数

- `--line-cruise-vx`: 正常行内巡航速度
- `--lidar-row-entry-*`: 换行后的停车识别和低速入行参数
- `--lidar-*`: 激光雷达局部中线跟踪参数

## 当前实现说明

- 局部控制由进程内的 `plant_lidar_centerline_follower.py` 提供
- 正常行内使用激光控制；任务过渡阶段使用全局路径控制
- 最终底盘 CAN 命令由 `autorunlida` 统一下发
