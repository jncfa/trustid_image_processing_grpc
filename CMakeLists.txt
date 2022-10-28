# Copyright 2018 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cmake "superbuild" file for C++ helloworld example.
# This build file demonstrates how to build the helloworld project
# and all its dependencies in a single cmake build (hence "superbuild")
# that is easy to build and maintain.
# cmake's ExternalProject_Add() is used to import all the sub-projects,
# including the "helloworld" project itself.
# See https://blog.kitware.com/cmake-superbuilds-git-submodules/

cmake_minimum_required(VERSION 3.5.1)

# Project
project(trustid-image-processing-grpc C CXX)

include(FetchContent)
FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.50.0
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(gRPC)

add_subdirectory("../trustid_image_processing_win" "trustid_image_processing_win")

get_filename_component(hw_proto "trustid.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/TRUSTID.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/TRUSTID.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/TRUSTID.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/TRUSTID.grpc.pb.h")

add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND protobuf::protoc
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc=/usr/local/bin/grpc_cpp_plugin
        "${hw_proto}"
      DEPENDS "${hw_proto}")

add_executable(trustid-image-processing-grpc-simpleserver "grpc-simpleserver.cc" ${hw_proto_srcs} ${hw_grpc_srcs})
target_include_directories(trustid-image-processing-grpc-simpleserver PUBLIC "includes" ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries( trustid-image-processing-grpc-simpleserver PUBLIC dlib trustid-image-processing-lib ${OpenCV_LIBS} grpc++_reflection gpr grpc grpc++ protobuf::libprotoc protobuf::libprotobuf )

add_executable(trustid-image-processing-grpc-simpleclient "grpc-simpleclient.cc" ${hw_proto_srcs} ${hw_grpc_srcs})
target_include_directories(trustid-image-processing-grpc-simpleclient PUBLIC "includes" ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries( trustid-image-processing-grpc-simpleclient PUBLIC dlib trustid-image-processing-lib dlib ${OpenCV_LIBS} grpc++ grpc++_reflection gpr grpc protobuf::libprotoc protobuf::libprotobuf )
#set(OpenCV_DIR ../trustid_image_processing_win/depends/opencv/build)
#find_package(OpenCV REQUIRED )

if(WIN32)

# Copy opencv_world.dll to the same directory as the executable
#get_target_property(_opencv_dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
#get_target_property(_opencv_dll_release opencv_world  IMPORTED_LOCATION_RELEASE)

#add_custom_command(TARGET trustid-image-processing-grpc-simpleserver POST_BUILD
#COMMAND ${CMAKE_COMMAND} -E copy_if_different           # which executes "cmake - E copy_if_different..."
#"$<$<CONFIG:debug>:${_opencv_dll_dbg}>$<$<CONFIG:release>:${_opencv_dll_release}>"      # <--this is in-file
#$<TARGET_FILE_DIR:trustid-image-processing-grpc-simpleserver>                        # <--this is out-file path
#)
# Copy zlib.dll to the same directory as the executable
# get_filename_component( ZLIB_LIBDIR "${ZLIB_LIBRARY}" PATH )
# get_filename_component ( ZLIB_BASENAME "${ZLIB_LIBRARY}" NAME_WE )
# get_filename_component ( ZLIB_LIBDIR_BASE "${ZLIB_LIBDIR}" PATH )

# find_file ( ZLIB_DLL
# "${CMAKE_SHARED_LIBRARY_PREFIX}${ZLIB_BASENAME}${CMAKE_SHARED_LIBRARY_SUFFIX}"
#   HINTS
#     "${ZLIB_LIBDIR_BASE}"
#   PATH_SUFFIXES
#     bin
#   NO_DEFAULT_PATH
# )

# add_custom_command(TARGET trustid-image-processing-grpc-simpleserver POST_BUILD
# COMMAND ${CMAKE_COMMAND} -E copy_if_different           # which executes "cmake - E copy_if_different..."
# "$<$<CONFIG:debug>:${ZLIB_DLL}>$<$<CONFIG:release>:${ZLIB_DLL}>"      # <--this is in-file
# $<TARGET_FILE_DIR:trustid-image-processing-grpc-simpleserver>                        # <--this is out-file path
# )

endif(WIN32)

