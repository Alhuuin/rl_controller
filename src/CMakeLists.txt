# ---- ONNX Runtime local setup ----

# Absolute paths to your local ONNX Runtime
set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)
# For build-time runs (ctest, running from build dir)
set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

# For after install
set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Find *only* in the local lib dir, never system paths
find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime
  PATHS ${ONNXRUNTIME_LIB_DIR}
  NO_DEFAULT_PATH
  REQUIRED
)

if(NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_LIB_DIR}")
endif()

message(STATUS "Using ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Using ONNX Runtime lib: ${ONNXRUNTIME_LIB}")

# ---- Sources ----

set(SRC
  RLController.cpp
  RLPolicyInterface.cpp
  utils.cpp
  PolicySimulatorHandling.cpp
)

set(HDR
  RLController.h
  RLPolicyInterface.h
  utils.h
  PolicySimulatorHandling.h
)

# ---- Main controller library ----

add_library(${PROJECT_NAME} SHARED ${SRC} ${HDR})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DRLCONTROLLER_EXPORTS")
target_link_libraries(${PROJECT_NAME} PUBLIC mc_rtc::mc_control_fsm)

# Add mc_joystick_plugin include directory if available (header-only)
find_path(MC_JOYSTICK_INCLUDE_DIR
  NAMES mc_joystick_plugin/joystick_inputs.h
  HINTS ${CMAKE_INSTALL_PREFIX} ENV CMAKE_PREFIX_PATH
  PATH_SUFFIXES include
)

if(MC_JOYSTICK_INCLUDE_DIR)
  target_include_directories(${PROJECT_NAME} PUBLIC ${MC_JOYSTICK_INCLUDE_DIR})
  message(STATUS "Found mc_joystick_plugin headers at ${MC_JOYSTICK_INCLUDE_DIR}")
else()
  message(STATUS "mc_joystick_plugin headers not found - joystick functionality will be unavailable")
endif()

# ---- Plugin target (mc_rtc controller entry point) ----

add_controller(${PROJECT_NAME}_controller lib.cpp "")
target_link_libraries(${PROJECT_NAME}_controller PUBLIC ${PROJECT_NAME})

# ---- Link ONNX to both targets ----

foreach(tgt IN ITEMS ${PROJECT_NAME} ${PROJECT_NAME}_controller)
  target_include_directories(${tgt} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(${tgt} PRIVATE ${ONNXRUNTIME_LIB})
endforeach()

# ---- Install ----

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION "${MC_RTC_LIBDIR}"
  LIBRARY DESTINATION "${MC_RTC_LIBDIR}"
  RUNTIME DESTINATION bin
)

# ---- Controller state machine ----

add_subdirectory(states)