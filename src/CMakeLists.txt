set(CONTROLLER_NAME RLController)

# ---- ONNX Runtime (local vendored copy) ----
set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)

find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime
  PATHS ${ONNXRUNTIME_LIB_DIR}
  NO_DEFAULT_PATH
  REQUIRED
)

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
  api.h
)

add_subdirectory(states)

add_library(${CONTROLLER_NAME} SHARED ${SRC} ${HDR})

target_link_libraries(${CONTROLLER_NAME}
  PUBLIC
    mc_rtc::mc_control_fsm
    ${ONNXRUNTIME_LIB}
)

target_include_directories(${CONTROLLER_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${ONNXRUNTIME_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RLController>
)

add_controller(${CONTROLLER_NAME}_controller lib.cpp "")
target_link_libraries(${CONTROLLER_NAME}_controller PUBLIC ${CONTROLLER_NAME})

install(
  TARGETS ${CONTROLLER_NAME}
  DESTINATION ${MC_RTC_LIBDIR}
  EXPORT ${TARGETS_EXPORT_NAME}
)

install(
  FILES RLController.h api.h PolicySimulatorHandling.h utils.h RLPolicyInterface.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RLController
)
