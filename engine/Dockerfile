FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base
WORKDIR /cusymint

COPY . ./engine-src

RUN apt-get update && apt-get install -y \
    cmake \
    libfmt-dev

RUN cmake -S engine-src -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build -j$((1 + $(nproc)))
RUN build/testcusymint
