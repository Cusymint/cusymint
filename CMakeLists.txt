cmake_minimum_required(VERSION 3.7)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cusymint LANGUAGES CXX CUDA)

set(SRC_DIR "src")

add_executable(
  cusymint
  ${SRC_DIR}/main.cu
  ${SRC_DIR}/cuda_utils.cu
  ${SRC_DIR}/symbol.cu
  ${SRC_DIR}/expander_placeholder.cu
  ${SRC_DIR}/unknown.cu
  ${SRC_DIR}/constants.cu
  ${SRC_DIR}/variable.cu
  ${SRC_DIR}/addition.cu
  ${SRC_DIR}/product.cu
  ${SRC_DIR}/power.cu
  ${SRC_DIR}/trigonometric.cu
  ${SRC_DIR}/integrate.cu
  ${SRC_DIR}/integral.cu
  ${SRC_DIR}/substitution.cu
  ${SRC_DIR}/solution.cu
  ${SRC_DIR}/inverse_trigonometric.cu)

target_compile_features(cusymint PUBLIC cxx_std_14)
set_target_properties(cusymint PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                          CUDA_ARCHITECTURES OFF)

if(APPLE)
  set_property(TARGET cusymint PROPERTY BUILD_RPATH
                                        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
