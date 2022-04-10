cmake_minimum_required(VERSION 3.7)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cusymint LANGUAGES CXX CUDA)

add_executable(
  cusymint
  src/main.cu
  src/symbol.cu
  src/constants.cu
  src/variable.cu
  src/addition.cu
  src/product.cu
  src/power.cu
  src/trigonometric.cu
  src/integrate.cu)

target_compile_features(cusymint PUBLIC cxx_std_14)
set_target_properties(cusymint PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                          CUDA_ARCHITECTURES OFF)

if(APPLE)
  set_property(TARGET cusymint PROPERTY BUILD_RPATH
                                        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
