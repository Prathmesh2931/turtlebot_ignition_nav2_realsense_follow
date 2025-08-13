# navigation2_ignition_gazebo_turtlebot3
Using Nav2 for navigating a simulated Turtlebot 3 in the Ignition Gazebo simulator.

## Basic Navigation

`ROS_LOCALHOST_ONLY=1 TURTLEBOT3_MODEL=waffle ros2 launch turtlebot3 simulation.launch.py` to launch simulation, nav2, and rviz2 simultaneously.

## Red Box Following

This package now includes red box detection and following capabilities using the RealSense camera.

### Launch with Red Box Following

To launch the simulation with red box detection and following enabled:

```bash
ROS_LOCALHOST_ONLY=1 TURTLEBOT3_MODEL=waffle ros2 launch turtlebot3 simulation.launch.py enable_follower:=true
```

This will:
1. Launch the Ignition Gazebo simulation
2. Spawn the TurtleBot3 with RealSense RGB and depth cameras
3. Spawn a red box in the simulation
4. Start the navigation stack with custom RViz configuration
5. Start the red box follower node

### Manual Red Box Follower Launch

You can also launch the red box follower separately:

```bash
ros2 launch turtlebot3 red_box_follower.launch.py
```

### Features

- **Enhanced RealSense Camera**: Includes both RGB and depth camera sensors
- **Red Box Detection**: Computer vision-based detection using HSV color filtering
- **Distance Estimation**: Uses depth camera for accurate distance measurement
- **Autonomous Following**: Robot automatically follows the detected red box
- **Safety Features**: Maintains safe following distance and stops when box is not detected
- **Custom RViz Config**: Enhanced visualization showing camera feeds and depth information

### Camera Topics

- `/camera/rgb/image_raw` - RGB camera feed
- `/camera/rgb/camera_info` - RGB camera info
- `/camera/depth/image_raw` - Depth camera feed  
- `/camera/depth/camera_info` - Depth camera info

### Control Parameters

The red box follower can be tuned by modifying parameters in `src/red_box_follower.py`:

- `target_distance`: Desired following distance (default: 1.5m)
- `distance_tolerance`: Distance control tolerance (default: 0.3m)
- `angular_gain`: Proportional gain for turning (default: 0.8)
- `linear_gain`: Proportional gain for forward/backward motion (default: 0.5)
- `max_linear_vel`: Maximum forward/backward speed (default: 0.3 m/s)
- `max_angular_vel`: Maximum turning speed (default: 0.8 rad/s)

![Turtlebot3 screen shot](./docs/media/turtlebot3scr.png)

## Technical Details

`/odom` topic, `odom` frame, and `/odom/tf` (tf topic) are defined in `model.sdf`.  The transformation of `base_footprint` in the odom frame is published through `/odom/tf`.  `/odom` topic publishes the transformation of odom frame in the `map` frame.

Use `ros2 run tf2_tools view_frames` to see the tf frame relations.

Ign gazebo publishes `joint_states`, which is then translated to ROS2 topic via `ros_ign_bridge`, and consumed by `robot_state_publisher` (a ROS2 node) for computing/publishing most of tf.

`/odom/tf` is remapped to `/tf`.

Ign gazebo topics are translated to/from ROS2 topics via `ros_ign_bridge`.

`nav2_bringup` is called to initiate basic services and configurations.

Tested with Ignition Gazebo Fortress and ROS2 Humble.

## Requirements

- `ros-<distro>-navigation2`
- `ros-<distro>-nav2-bringup`
- `ros-<distro>-ros-ign-gazebo`
- `ros-<distro>-ros-ign-bridge`
- `ros-<distro>-cv-bridge`
- `ros-<distro>-opencv-python`
