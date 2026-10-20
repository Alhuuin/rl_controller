cmake_minimum_required(VERSION 3.22)

set(PROJECT_NAME rl_controller)
set(PROJECT_DESCRIPTION "RL Policy FSM Controller")
set(PROJECT_URL https://github.com/Alhuuin/rl_controller)

set(PROJECT_USE_CMAKE_EXPORT TRUE)

set(CMAKE_CXX_STANDARD 17)
set(CXX_DISABLE_WERROR ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

find_package(jrl-cmakemodules QUIET CONFIG)
if(jrl-cmakemodules_FOUND)
  get_property(
    JRL_CMAKE_MODULES
    TARGET jrl-cmakemodules::jrl-cmakemodules
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  )
  message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
else()
  message(STATUS "JRL cmakemodules not found. Let's fetch it.")
  include(FetchContent)
  FetchContent_Declare(
    "jrl-cmakemodules"
    GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git"
    GIT_TAG c32298a98e736ac4d15a9540dcd1f22cd7f161db
  )
  FetchContent_MakeAvailable("jrl-cmakemodules")
  FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
endif()
include(${JRL_CMAKE_MODULES}/base.cmake)

project(${PROJECT_NAME} LANGUAGES CXX)

###########
# OPTIONS #
###########
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime" CACHE PATH "Path to ONNX Runtime root directory")


add_project_dependency(mc_rtc REQUIRED)
add_project_dependency(mc_joystick_plugin REQUIRED)

if(NOT TARGET mc_rtc::mc_control_fsm)
  add_project_dependency(mc_rtc REQUIRED)
endif()

add_subdirectory(src)

set(ONNX_POLICY_DIR "${MC_CONTROLLER_RUNTIME_INSTALL_PREFIX}/share/rl_controller/policy")
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/policy/
  DESTINATION ${ONNX_POLICY_DIR} 
  FILES_MATCHING PATTERN "*.onnx"
)

configure_file(
  etc/RLController.in.yaml
  "${CMAKE_CURRENT_BINARY_DIR}/etc/RLController.yaml"
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/etc/RLController.yaml"
  DESTINATION "${MC_CONTROLLER_RUNTIME_INSTALL_PREFIX}/etc"
)


