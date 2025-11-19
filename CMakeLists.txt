cmake_minimum_required(VERSION 3.10)

set(CXX_DISABLE_WERROR 1)
set(CMAKE_CXX_STANDARD 17)

set(PROJECT_NAME RLController)
set(PROJECT_DESCRIPTION "RL Policy FSM Controller for H1 Robot")
set(PROJECT_URL "")

add_compile_definitions(
  PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

project(
  ${PROJECT_NAME}
  LANGUAGES CXX
  VERSION 1.0.0)
include(CTest)
enable_testing()

# Check if the project is built inside mc_rtc
if(NOT TARGET mc_rtc::mc_control)
  find_package(mc_rtc REQUIRED)
endif()

add_subdirectory(src)

# if(BUILD_TESTING)
#   add_subdirectory(tests)
# endif()

configure_file(etc/RLController.in.yaml
               "${CMAKE_CURRENT_BINARY_DIR}/etc/RLController.yaml")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/etc/RLController.yaml"
        DESTINATION "${MC_CONTROLLER_RUNTIME_INSTALL_PREFIX}/etc")