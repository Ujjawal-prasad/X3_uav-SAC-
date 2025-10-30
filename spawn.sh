gnome-terminal -- bash -c "cd && cd <path_to_model.sdf> && QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins && gz sim model.sdf; exec bash"
gnome-terminal -- bash -c "ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=ros_gz_bridge.yaml"
