FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS build
WORKDIR /cusymint

COPY . ./engine-src

RUN apt-get update && apt-get install -y \
    cmake \
    git

RUN cmake -S engine-src -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build -j$((1 + $(nproc)))


FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS runtime
WORKDIR /cusymint

COPY --from=build /cusymint/build/srvcusymint .

EXPOSE 8000
ENTRYPOINT [ "/cusymint/srvcusymint", "0.0.0.0:8000" ]
