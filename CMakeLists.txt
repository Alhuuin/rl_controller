cmake_minimum_required(VERSION 3.22)

set(PROJECT_NAME rl_controller)
set(PROJECT_DESCRIPTION "RL Policy FSM Controller")
set(PROJECT_URL https://github.com/Alhuuin/rl_controller)

set(PROJECT_USE_CMAKE_EXPORT TRUE)

set(CMAKE_CXX_STANDARD 17)
set(CXX_DISABLE_WERROR ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

include(cmake/base.cmake)

project(${PROJECT_NAME} LANGUAGES CXX)

add_project_dependency(mc_rtc REQUIRED)

if(NOT TARGET mc_rtc::mc_control_fsm)
  find_package(mc_rtc REQUIRED)
endif()

add_subdirectory(src)

configure_file(
  etc/RLController.in.yaml
  "${CMAKE_CURRENT_BINARY_DIR}/etc/RLController.yaml"
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/etc/RLController.yaml"
  DESTINATION "${MC_CONTROLLER_RUNTIME_INSTALL_PREFIX}/etc"
)
