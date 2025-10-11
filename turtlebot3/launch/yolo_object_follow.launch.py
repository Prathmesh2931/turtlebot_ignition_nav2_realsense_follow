import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

# TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']
TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'waffle')


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_name = LaunchConfiguration('world_name', default='turtlebot3_world')

    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3'), 'launch')
    rviz_config_dir = os.path.join(get_package_share_directory('turtlebot3'), 'rviz', 'rviz.rviz')

    ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            os.path.join("/opt/ros/humble", "share"),
            ":" + os.path.join(get_package_share_directory('turtlebot3'), "models")
        ]
    )

    # Spawn robot
    ignition_spawn_entity = Node(
        package='ros_ign_gazebo',
        executable='create',
        output='screen',
        arguments=[
            '-entity', TURTLEBOT3_MODEL,
            '-name', TURTLEBOT3_MODEL,
            '-file', PathJoinSubstitution([
                get_package_share_directory('turtlebot3'),
                "models", "turtlebot3", "model.sdf"
            ]),
            '-allow_renaming', 'true',
            '-x', '-2.0',
            '-y', '-0.5',
            '-z', '0.01'
        ]
    )

    # Spawn world
    ignition_spawn_world = Node(
        package='ros_ign_gazebo',
        executable='create',
        output='screen',
        arguments=[
            '-file', PathJoinSubstitution([
                get_package_share_directory('turtlebot3'),
                "models", "worlds", "model.sdf"
            ]),
            '-allow_renaming', 'false'
        ]
    )

    world_only = os.path.join(
        get_package_share_directory('turtlebot3'),
        "models", "worlds", "world_only.sdf"
    )

    # YOLO Object Tracker Node
    yolo_tracker_node = Node(
        package='object_tracking',
        executable='yolo_object_tracker.py',
        name='yolo_object_tracker',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Voice Command Interface Node
    voice_interface_node = Node(
        package='object_tracking',
        executable='voice_cmd_interface.py',
        name='voice_cmd_interface',
        output='screen'
    )

    # RViz node with delay
    delayed_rviz = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config_dir],
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        ign_resource_path,
        ignition_spawn_entity,
        ignition_spawn_world,
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('ros_ign_gazebo'),
                             'launch', 'ign_gazebo.launch.py')
            ]),
            launch_arguments=[('ign_args', [' -r -v 3 ' + world_only])]),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/ros_ign_bridge.launch.py']),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/robot_state_publisher.launch.py']),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        yolo_tracker_node,
        voice_interface_node,
        delayed_rviz,

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),

        DeclareLaunchArgument(
            'world_name',
            default_value=world_name,
            description='World name'),
    ])