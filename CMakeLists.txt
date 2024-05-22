cmake_minimum_required(VERSION 3.5)
project(tvm_inference_server LANGUAGES CXX CUDA)

add_definitions(-w) # Ignore warnings
add_definitions(-D__MISSILE_CUDA_GPU__)

set(RESOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/resource" )
add_definitions(-DRESOURCE_DIR="${RESOURCE_DIR}")

SET(CMAKE_BUILD_TYPE "Debug")

add_definitions(-DDEBUG)
add_definitions("$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -std=c++14")

add_library(gdrapi SHARED IMPORTED GLOBAL)
set_target_properties(gdrapi PROPERTIES IMPORTED_LOCATION /usr/local/gdrcopy/lib/libgdrapi.so)

LIST(APPEND CUDA_NVCC_FLAGS
        "-gencode;arch=compute_86,code=sm_86;
        -gencode;arch=compute_70,code=sm_70;
        -gencode;arch=compute_61,code=sm_61;
        -v;
        -std=c++14;"
        ) # Set the target gpu architecture

if(UNIX OR APPLE)
    MESSAGE(STATUS "DETECT OS ENVIRONMENT IS UNIX/APPLE")
    LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC")
else(
    MESSAGE(STATUS "DETECT OS ENVIRONMENT IS NOT UNIX/APPLE")
)
endif()

SET(HOME_TVM "tvm_shared")
include(cmake/Util.cmake)
include(cmake/FindCUDA.cmake)
set(USE_CUDA ON)
find_cuda(${USE_CUDA})

find_package(CUDA  REQUIRED)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Boost REQUIRED COMPONENTS stacktrace_basic stacktrace_backtrace stacktrace_addr2line stacktrace_noop)

MESSAGE(STATUS "CUDA_NVCC_FLAGS is " ${CUDA_NVCC_FLAGS})

#### For grpc ####

include("${CMAKE_CURRENT_LIST_DIR}/cmake/common.cmake")
get_filename_component(missile_proto "${CMAKE_CURRENT_LIST_DIR}/src/protos/missile.proto" ABSOLUTE)
get_filename_component(missile_proto_path "${missile_proto}" PATH)
set(GRPC_GENERATE_DIR "${PROJECT_SOURCE_DIR}/src/rpc" )
set(missile_proto_srcs "${GRPC_GENERATE_DIR}/missile.pb.cc")
set(missile_proto_hdrs "${GRPC_GENERATE_DIR}/missile.pb.h")
set(missile_grpc_srcs "${GRPC_GENERATE_DIR}/missile.grpc.pb.cc")
set(missile_grpc_hdrs "${GRPC_GENERATE_DIR}/missile.grpc.pb.h")
add_custom_command(
        OUTPUT "${missile_proto_srcs}" "${missile_proto_hdrs}" "${missile_grpc_srcs}" "${missile_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${GRPC_GENERATE_DIR}"
        --cpp_out "${GRPC_GENERATE_DIR}"
        -I "${missile_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${missile_proto}"
        DEPENDS "${missile_proto}")

include_directories("${GRPC_GENERATE_DIR}")

MESSAGE(STATUS "_PROTOBUF_PROTOC is " ${_PROTOBUF_PROTOC})

# missile_grpc_proto
add_library(missile_grpc_proto
        ${missile_grpc_srcs}
        ${missile_grpc_hdrs}
        ${missile_proto_srcs}
        ${missile_proto_hdrs})
target_link_libraries(missile_grpc_proto
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})

#### For sample cuda kernel code ####

function (add_cuda_kernel_ptx program_name program_path)
    add_custom_command(
        OUTPUT "${program_path}/${program_name}.ptx"
        COMMAND nvcc
        ARGS -ptx
            -arch "sm_75"
            --compiler-bindir "/bin/clang"
            "${program_path}/${program_name}.cu"
    )
endfunction()


#### For server & client ####
include_directories("${PROJECT_SOURCE_DIR}/src")
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/client" client)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/executor" executor)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/cuda_implementation" cuda_impl)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/executor/cuda" missile)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/executor/temporal_multiplexing" temporal_multiplexing)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/executor/spatial_multiplexing" spatial_multiplexing)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/executor/mpsplus_multiplexing" mpsplus_multiplexing)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/executor/naive" mps_multiplexing)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/server" server)
#AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/server/missile_temporal" missile_temporal_scheduler)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/server/temporal_multiplexing" temporal_multiplexing_scheduler)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/server/spatial_multiplexing" spatial_multiplexing_scheduler)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/server/mps_plus" mps_plus_scheduler)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/server/naive" mps_multiplexing_scheduler)
AUX_SOURCE_DIRECTORY("${PROJECT_SOURCE_DIR}/src/util" util)

add_library(missile_util
    ${util}
)

add_library(missile_server_lib
    ${server}
    ${executor}
    ${cuda_impl}
    ${missile}
    ${missile_temporal_scheduler}
    ${temporal_multiplexing}
    ${temporal_multiplexing_scheduler}
    ${spatial_multiplexing}
    ${spatial_multiplexing_scheduler}
    ${mps_multiplexing}
    ${mps_multiplexing_scheduler}
    ${mpsplus_multiplexing}
    ${mps_plus_scheduler}
)

add_library(missile_client_lib
    ${client}
)

function (add_executable_app app_name app_path)
    cuda_add_executable(${app_name}
            ${app_path}
            )
    target_link_libraries(${app_name}
            missile_server_lib
            missile_client_lib
            missile_util
            missile_grpc_proto
            gdrapi
            ${GTEST_BOTH_LIBRARIES}
            ${_REFLECTION}
            ${_GRPC_GRPCPP}
            ${_PROTOBUF_LIBPROTOBUF}
            glog::glog
            cudart
            nvrtc
            cuda
            pthread
            m
            nvidia-ml
            boost_stacktrace_backtrace
            dl
            backtrace
            )
endfunction()

add_executable_app(missile_client "${PROJECT_SOURCE_DIR}/src/example/rpc_client.cpp")
add_executable_app(missile_client_cont "${PROJECT_SOURCE_DIR}/src/example/rpc_client_cont.cpp")
add_executable_app(missile_client_poisson "${PROJECT_SOURCE_DIR}/src/example/rpc_client_poisson.cpp")
add_executable_app(missile_client_trace "${PROJECT_SOURCE_DIR}/src/example/rpc_client_trace.cpp")
add_executable_app(missile_server "${PROJECT_SOURCE_DIR}/src/example/rpc_server.cpp")
