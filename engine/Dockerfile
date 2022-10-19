ENV CUDA_VERSION="11.8.0"
ENV UBUNTU_VERSION="22.04"

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base
WORKDIR /cusymint

COPY . ./engine-src

RUN apt update
RUN apt install -y libfmt-dev

RUN cmake -S engine-src -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --target install