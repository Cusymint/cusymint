# Cusymint

Symbolic integration run in parallel on Nvidia GPUs. Alternative to Wolfram|Alpha or Mathematica.

# Features
The app solves integrals and provides a helpful list of integral examples for learning syntax.

https://user-images.githubusercontent.com/62249621/234692468-6d0bc7c8-ee2b-4c03-8430-e4a4ea55c232.mov

Outline the steps by which the engine arrived at a solution

https://user-images.githubusercontent.com/62249621/234692818-b9db86fd-fc32-41e6-bae9-fc6277370635.mov

User can browse local history and modify it.

https://user-images.githubusercontent.com/62249621/234693011-e67da68f-6289-4039-91a4-bb04c582353d.mov

# Installation
There are many ways in which cusymint can be installed and run. It consists of two parts:
- engine (or backend) which needs to be run on a computer with Nvidia GPU,
- client which can be run on pretty much any computer or mobile device.

Cusymint is meant to be self-hosted, so you can run it on your own computer or server. The easiest way to get started is to use the Docker image. If you want to run cusymint on your own computer, you can use the pre-built binaries. If you want to build cusymint from source, you can do that too.

> Note that since cusymint is self-hosted knowing IP address of the computer running the engine is needed. 
You can change default IP address in the client settings.

## Docker
The easiest way to get started is to use the Docker image. You can find the Docker images on [Github packages](https://github.com/orgs/Cusymint/packages?repo_name=cusymint).
```bash
# Pull the engine image
docker pull ghcr.io/cusymint/cusymint/engine:latest
# Pull the client image
docker pull ghcr.io/cusymint/cusymint/client:latest
```
To run cusymint, you need to have Docker installed. You can find instructions on how to install Docker [here](https://docs.docker.com/get-docker/).
Depending on your OS you may also need to install Nvidia Container Toolkit. You can find instructions on how to install Nvidia Container Toolkit [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
Note that when running Docker on WSL2 you can probably skip the Nvidia container toolkit installation ([see reference](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)).
Once you have Docker installed, you can run cusymint with the following command:

```bash
# Run the engine, which will listen on port 8000
# You need to have Nvidia GPU and latest Nvidia drivers installed
docker run -d --gpus=all -p 8000:8000 ghcr.io/cusymint/cusymint/engine:latest
# Run the client, which will run on port 80
docker run -d -p 80:80 ghcr.io/cusymint/cusymint/client:latest
```
After that you should be able to access cusymint at http://localhost:80.

### Docker compose
You can also use docker-compose to run cusymint. You can find the docker-compose file in the [repository](https://github.com/cusymint/cusymint/blob/master/docker-compose.ghcr.yml).

To run cusymint with docker-compose, you need to have docker-compose installed. You can find instructions on how to install docker-compose [here](https://docs.docker.com/compose/install/). Once you have docker-compose installed, you can run cusymint with the following command:
```bash
# Run cusymint with docker-compose
# note that you need to have Nvidia GPU and latest Nvidia drivers installed
docker compose -f ./docker-compose.ghrc.yml up
```

## Pre-built binaries
You can download pre-built binaries for cusymint from the [releases page](todo). The binaries are available for Linux, Windows and Android.

### Engine
Download the engine binary for your platform from the [releases page](todo). You need to have Nvidia GPU and latest Nvidia drivers installed as well as CUDA 12 or later installed. See CUDA installation instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
You can run the binary with the following command:
```bash
# Run the engine, default port is 8000
./srvcusymint
# Run the engine, which will listen on port 9000
./srvcusymint ws://localhost:9000
```

### Client
Download the client binary for your platform from the [releases page](todo). You can run the client on any device, but it has been tested only on Ubuntu 22.04, Windows 11 and Android 11. When running as a web app, it has been tested on Chrome. Firefox seems to have some issues with the web app.

To install apk on Android follow the latest instructions [here](https://www.google.com/search?q=how+to+install+apk+on+android).

Client for Windows is packed as a zip file. You can extract it anywhere and run the executable named `cusymint_app.exe`.

Client for Linux is packed as a tar.gz file. You can extract it anywhere and run the executable named `cusymint_app`.

## Build from source
You can build cusymint from source.

### Engine
To build the engine, you need to have the following installed:
- nvcc (Nvidia CUDA compiler),
- cmake,
- git.

To build the engine, run the following commands:
```bash
# Clone the repository
git clone https://github.com/cusymint/cusymint.git
# Go to the engine directory
cd cusymint/engine
# Specify cmake directories
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
# Build the engine
cmake --build build
```
The engine will be built in the `build` directory.

### Client
To build the client, you need to have the following installed:
- [Flutter](https://docs.flutter.dev/get-started/install) with chosen platform support,
- [melos](https://github.com/invertase/melos),
- git.

To build the client, run the following commands:
```bash
# Clone the repository
git clone https://github.com/cusymint/cusymint.git
# Bootstrap project
melos bootstrap
# Build the client
# where XXX is either android, linux, windows or web
melos run build:XXX
```
