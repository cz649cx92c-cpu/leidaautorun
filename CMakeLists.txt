cmake_minimum_required(VERSION 3.8)
project(autorun_final)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

install(PROGRAMS
  main.py
  backend.py
  app.py
  uvc_preview_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  package.xml
  README.md
  gui_settings.json
  run_cli.sh
  run_gui.sh
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  missions
  maps
  vendor
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
