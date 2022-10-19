FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base
WORKDIR /cusymint

COPY . ./engine-src

RUN apt-get update && apt-get install -y libfmt-dev cmake

RUN cmake -S engine-src -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build
