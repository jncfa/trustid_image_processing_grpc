# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version (use Debian 11, Ubuntu 18.04/22.04 on local arm64/Apple Silicon): debian-11, debian-10, ubuntu-22.04, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

# [Optional] Install CMake version different from what base image has already installed. 
# CMake reinstall choices: none, 3.21.5, 3.22.2, or versions from https://cmake.org/download/
ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.24.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/
RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# Clone OpenCV (+ contrib repo)
#RUN git clone -b 4.6.0 --single-branch https://github.com/opencv/opencv.git \
#    && git clone -b 4.6.0 --single-branch https://github.com/opencv/opencv_contrib.git \
#    && mkdir opencvbuild && cd opencvbuild \
#    && cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -D CMAKE_BUILD_TYPE=RELEASE ../opencv \
#    && cmake --build . --config Release -- -j8 \
#    && cmake --install .

# Clone gRPC repo & subdependencies, build, and install
#RUN git clone -b v1.50.0 https://github.com/grpc/grpc && cd grpc \
#    && git submodule update --init \
#    && mkdir grpcbuild && cd grpcbuild \ 
#    && cmake .. \
#        -DCMAKE_BUILD_TYPE=Release \
#        -DgRPC_INSTALL=ON \
#        -DgRPC_BUILD_TESTS=OFF \
#        -DgRPC_SSL_PROVIDER=package \ 
#    && cmake --build . --config Release -- -j8  \
#    && cmake --install .

# Install gRPC C++ dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends build-essential autoconf libtool pkg-config g++ wget unzip libgrpc libopencv-dev libgrpc++-dev

EXPOSE 50051
