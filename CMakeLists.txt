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

# Note: For all external projects, instead of using checked-out code, one could
# specify GIT_REPOSITORY and GIT_TAG to have cmake download the dependency directly,
# without needing to add a submodule to your project.

# # Builds zlib project from the git submodule.
# ExternalProject_Add(zlib
#   PREFIX zlib
#   SOURCE_DIR "C:/Users/jncfa/source/grpc/third_party/zlib"
#   CMAKE_CACHE_ARGS
#         -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/zlib
# )
#add_subdirectory("C:/Users/jncfa/source/grpc/third_party/zlib" zlib)

# Builds absl project from the git submodule.
# ExternalProject_Add(absl
#   PREFIX absl
#   SOURCE_DIR "C:/Users/jncfa/source/grpc/third_party/abseil-cpp"
#   CMAKE_CACHE_ARGS
#         -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
#         -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/absl
# )
#add_subdirectory("C:/Users/jncfa/source/grpc/third_party/abseil-cpp" abseil-cpp)

# # Builds c-ares project from the git submodule.
# ExternalProject_Add(c-ares
#   PREFIX c-ares
#   SOURCE_DIR "C:/Users/jncfa/source/grpc/third_party/cares/cares"
#   CMAKE_CACHE_ARGS
#         -DCARES_SHARED:BOOL=OFF
#         -DCARES_STATIC:BOOL=ON
#         -DCARES_STATIC_PIC:BOOL=ON
#         -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares
# )
#add_subdirectory("C:/Users/jncfa/source/grpc/third_party/cares/cares" c-ares)

# # Builds protobuf project from the git submodule.
# ExternalProject_Add(protobuf
#   PREFIX protobuf
#   SOURCE_DIR "C:/Users/jncfa/source/grpc/third_party/protobuf/cmake"
#   CMAKE_CACHE_ARGS
#         -Dprotobuf_BUILD_TESTS:BOOL=OFF
#         -Dprotobuf_WITH_ZLIB:BOOL=OFF
#         -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
#         -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/protobuf
# )
#add_subdirectory("C:/Users/jncfa/source/grpc/third_party/protobuf" protobuf)

# # Builds re2 project from the git submodule.
# ExternalProject_Add(re2
#   PREFIX re2
#   SOURCE_DIR "C:/Users/jncfa/source/grpc/third_party/re2"
#   CMAKE_CACHE_ARGS
#         -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
#         -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/re2
# )
#add_subdirectory("C:/Users/jncfa/source/grpc/third_party/re2" re2)

# # the location where protobuf-config.cmake will be installed varies by platform
# if (WIN32)
#   set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/cmake")
# else()
#   set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/lib/cmake/protobuf")
# endif()

# # if OPENSSL_ROOT_DIR is set, propagate that hint path to the external projects with OpenSSL dependency.
# set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")
# if (OPENSSL_ROOT_DIR)
#   set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}")
# endif()

# # Builds gRPC based on locally checked-out sources and set arguments so that all the dependencies
# # are correctly located.
# ExternalProject_Add(grpc
#   PREFIX grpc
#   SOURCE_DIR "C:/Users/jncfa/source/grpc/"
#   CMAKE_CACHE_ARGS
#         -DgRPC_INSTALL:BOOL=ON
#         -DgRPC_BUILD_TESTS:BOOL=OFF
#         -DgRPC_BUILD_MSVC_MP_COUNT:STRING=-1
#         -DgRPC_PROTOBUF_PROVIDER:STRING=package
#         -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
#         -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
#         -DgRPC_RE2_PROVIDER:STRING=package
#         -Dre2_DIR:STRING=${CMAKE_CURRENT_BINARY_DIR}/re2/lib/cmake/re2
#         -DgRPC_ZLIB_PROVIDER:STRING=package
#         -DZLIB_ROOT:STRING=${CMAKE_CURRENT_BINARY_DIR}/zlib
#         -DgRPC_ABSL_PROVIDER:STRING=package
#         -Dabsl_DIR:STRING=${CMAKE_CURRENT_BINARY_DIR}/absl/lib/cmake/absl
#         -DgRPC_CARES_PROVIDER:STRING=package
#         -Dc-ares_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
#         -DgRPC_SSL_PROVIDER:STRING=package
#         ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
#         -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/grpc
#   DEPENDS c-ares protobuf re2 zlib absl
# )
#add_subdirectory("C:/Users/jncfa/source/grpc/" grpc)
set(GRPC_AS_SUBMODULE 1)
include(common.cmake)
# Build the helloworld projects itself using a CMakeLists.txt that assumes all the dependencies
# have already been installed.
# Even though helloworld is not really an "external project" from perspective of this build,
# we are still importing it using ExternalProject_Add because that allows us to use find_package()
# to locate all the dependencies (if we were building helloworld directly in this build we,
# we would have needed to manually import the libraries as opposed to reusing targets exported by
# gRPC and protobuf).

add_subdirectory("../trustid_image_processing_win" "trustid_image_processing_win")

get_filename_component(hw_proto "trustid.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/TRUSTID.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/TRUSTID.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/TRUSTID.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/TRUSTID.grpc.pb.h")
message(${_PROTOBUF_PROTOC}) 
message(${_GRPC_CPP_PLUGIN_EXECUTABLE}) 
message(${CMAKE_CURRENT_BINARY_DIR})
#get_target_property(grpc_cpp_plugin_location grpc_cpp_plugin LOCATION)

add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
      DEPENDS "${hw_proto}")

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_executable(trustid-image-processing-grpc-simpleserver "grpc-simpleserver.cc" ${hw_proto_srcs} ${hw_grpc_srcs})
target_include_directories(trustid-image-processing-grpc-simpleserver PUBLIC "includes" ${OpenCV_INCLUDE_DIRS}  ${hw_proto_hdrs} ${hw_grpc_hdrs})
target_link_libraries( trustid-image-processing-grpc-simpleserver PUBLIC dlib trustid-image-processing-lib ${OpenCV_LIBS} grpc++ grpc++_reflection gpr grpc grpc++ protobuf::libprotoc protobuf::libprotobuf )

add_executable(trustid-image-processing-grpc-simpleclient "grpc-simpleclient.cc" ${hw_proto_srcs} ${hw_grpc_srcs} "DeviceEnumerator.cpp")
target_include_directories(trustid-image-processing-grpc-simpleserver PUBLIC "includes" ${OpenCV_INCLUDE_DIRS} ${hw_proto_hdrs} ${hw_grpc_hdrs})
target_link_libraries( trustid-image-processing-grpc-simpleclient PUBLIC dlib trustid-image-processing-lib dlib ${OpenCV_LIBS} grpc++ grpc++_reflection gpr grpc protobuf::libprotoc protobuf::libprotobuf )
set(OpenCV_DIR ../trustid_image_processing_win/depends/opencv/build)
find_package(OpenCV REQUIRED )

# Copy opencv_world.dll to the same directory as the executable
get_target_property(_opencv_dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
get_target_property(_opencv_dll_release opencv_world  IMPORTED_LOCATION_RELEASE)

add_custom_command(TARGET trustid-image-processing-grpc-simpleserver POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different           # which executes "cmake - E copy_if_different..."
"$<$<CONFIG:debug>:${_opencv_dll_dbg}>$<$<CONFIG:release>:${_opencv_dll_release}>"      # <--this is in-file
$<TARGET_FILE_DIR:trustid-image-processing-grpc-simpleserver>                        # <--this is out-file path
)

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