cmake_minimum_required(VERSION 3.16)

if(NOT TARGET transfer_engine)
  message(FATAL_ERROR
          "Mooncake PG core requires the transfer_engine target (WITH_TE=ON)")
endif()

set(_pg_sources
    src/control_plane/agent.cpp
    src/control_plane/agent_host.cpp
    src/control_plane/coordinator.cpp
    src/control_plane/coordinator_host.cpp
    src/control_plane/link_manager.cpp
    src/control_plane/rpc_runtime.cpp
    src/mooncake_communicator.cpp
    src/mooncake_pg.cpp
    src/mooncake_worker_host.cpp
    src/mooncake_worker_thread.cpp
    src/p2p_proxy.cpp)

# The core implementation is framework-neutral, but its collective worker
# still has one device translation unit.  Compile that object directly with
# the platform compiler so the core does not need PyTorch's extension toolchain.
set(_pg_device_source "${CMAKE_CURRENT_SOURCE_DIR}/src/mooncake_worker.cu")
set(_pg_device_object "${CMAKE_CURRENT_BINARY_DIR}/mooncake_worker_device.o")
set(_pg_device_flags
    -std=c++20
    -O3
    --compiler-options=-fPIC
    "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../mooncake-transfer-engine/include")

if(USE_MUSA)
  set(_pg_musa_hints /usr/local/musa/bin)
  if(DEFINED ENV{MUSA_HOME})
    list(PREPEND _pg_musa_hints "$ENV{MUSA_HOME}/bin")
  endif()
  find_program(MOONCAKE_PG_DEVICE_COMPILER NAMES mcc
               HINTS ${_pg_musa_hints})
  if(NOT MOONCAKE_PG_DEVICE_COMPILER)
    message(FATAL_ERROR "USE_MUSA=ON requires the MUSA compiler (mcc)")
  endif()

  set(_pg_musa_source "${CMAKE_CURRENT_BINARY_DIR}/mooncake_worker.mu")
  list(APPEND _pg_device_flags
       -DUSE_MUSA
       -DMOONCAKE_EP_USE_MUSA=1
       --cuda-gpu-arch=mp_21
       --cuda-gpu-arch=mp_31)
  add_custom_command(
    OUTPUT "${_pg_device_object}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_pg_device_source}" "${_pg_musa_source}"
    COMMAND "${MOONCAKE_PG_DEVICE_COMPILER}" ${_pg_device_flags}
            -c "${_pg_musa_source}" -o "${_pg_device_object}"
    DEPENDS "${_pg_device_source}"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/mooncake_worker_kernels.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/pg_utils.h"
    COMMENT "Compiling the Mooncake PG MUSA worker"
    COMMAND_EXPAND_LISTS
    VERBATIM)
elseif(USE_MACA)
  set(_pg_maca_hints
      "${MACA_ROOT}/bin"
      "${MACA_ROOT}/cuda/bin"
      "${MACA_ROOT}/tools/cu-bridge/bin")
  if(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
    set(MOONCAKE_PG_DEVICE_COMPILER "$ENV{CUDACXX}")
  else()
    find_program(MOONCAKE_PG_DEVICE_COMPILER NAMES nvcc mxcc
                 HINTS ${_pg_maca_hints})
  endif()
  if(NOT MOONCAKE_PG_DEVICE_COMPILER)
    message(FATAL_ERROR
            "USE_MACA=ON requires CUDACXX or a MACA device compiler")
  endif()

  list(APPEND _pg_device_flags -DUSE_MACA -DMOONCAKE_EP_USE_MACA=1)
  add_custom_command(
    OUTPUT "${_pg_device_object}"
    COMMAND "${MOONCAKE_PG_DEVICE_COMPILER}" ${_pg_device_flags}
            -c "${_pg_device_source}" -o "${_pg_device_object}"
    DEPENDS "${_pg_device_source}"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/mooncake_worker_kernels.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/pg_utils.h"
    COMMENT "Compiling the Mooncake PG MACA worker"
    COMMAND_EXPAND_LISTS
    VERBATIM)
elseif(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(MOONCAKE_PG_DEVICE_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
  if(NOT MOONCAKE_PG_DEVICE_COMPILER)
    message(FATAL_ERROR "USE_CUDA=ON requires nvcc")
  endif()

  list(APPEND _pg_device_flags -DUSE_CUDA)
  if(TORCH_CUDA_ARCH_LIST)
    set(_pg_cuda_arches ${TORCH_CUDA_ARCH_LIST})
  else()
    set(_pg_cuda_arches 8.0 9.0)
  endif()
  foreach(_pg_arch IN LISTS _pg_cuda_arches)
    string(REPLACE "+PTX" "" _pg_arch_number "${_pg_arch}")
    string(REPLACE "." "" _pg_arch_number "${_pg_arch_number}")
    list(APPEND _pg_device_flags
         "-gencode=arch=compute_${_pg_arch_number},code=sm_${_pg_arch_number}")
    if(_pg_arch MATCHES "\\+PTX$")
      list(APPEND _pg_device_flags
           "-gencode=arch=compute_${_pg_arch_number},code=compute_${_pg_arch_number}")
    endif()
  endforeach()
  add_custom_command(
    OUTPUT "${_pg_device_object}"
    COMMAND "${MOONCAKE_PG_DEVICE_COMPILER}" ${_pg_device_flags}
            -c "${_pg_device_source}" -o "${_pg_device_object}"
    DEPENDS "${_pg_device_source}"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/mooncake_worker_kernels.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/pg_utils.h"
    COMMENT "Compiling the Mooncake PG CUDA worker"
    COMMAND_EXPAND_LISTS
    VERBATIM)
else()
  message(FATAL_ERROR
          "Mooncake PG currently requires one of USE_CUDA, USE_MUSA, or USE_MACA")
endif()

set_source_files_properties("${_pg_device_object}"
                            PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)

add_library(mooncake_pg SHARED ${_pg_sources} "${_pg_device_object}")

target_compile_features(mooncake_pg PRIVATE cxx_std_20)
target_include_directories(
  mooncake_pg
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)
target_link_libraries(mooncake_pg PRIVATE transfer_engine)

set_target_properties(
  mooncake_pg
  PROPERTIES C_VISIBILITY_PRESET hidden
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN YES
             BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN")

install(TARGETS mooncake_pg
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
install(FILES include/mooncake_pg.h DESTINATION include)
