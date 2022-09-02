cmake_minimum_required(VERSION 3.7)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cusymint LANGUAGES CXX CUDA)

set(SRC_DIR "src")
set(UTILS_DIR "${SRC_DIR}/Utils")
set(SYMBOL_DIR "${SRC_DIR}/Symbol")
set(EVALUATION_DIR "${SRC_DIR}/Evaluation")

add_executable(
  cusymint
  ${SRC_DIR}/main.cu
  ${UTILS_DIR}/Cuda.cu
  ${SYMBOL_DIR}/Symbol.cu
  ${SYMBOL_DIR}/ExpanderPlaceholder.cu
  ${SYMBOL_DIR}/Unknown.cu
  ${SYMBOL_DIR}/Constants.cu
  ${SYMBOL_DIR}/Variable.cu
  ${SYMBOL_DIR}/Addition.cu
  ${SYMBOL_DIR}/Product.cu
  ${SYMBOL_DIR}/Power.cu
  ${SYMBOL_DIR}/Trigonometric.cu
  ${SYMBOL_DIR}/Integral.cu
  ${SYMBOL_DIR}/Substitution.cu
  ${SYMBOL_DIR}/Solution.cu
  ${SYMBOL_DIR}/InverseTrigonometric.cu
  ${EVALUATION_DIR}/Integrate.cu)

target_include_directories(cusymint PRIVATE ${SRC_DIR})

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)

if(CMAKE_BUILD_TYPE_UPPER MATCHES DEBUG)
  target_compile_options(cusymint PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -g -G >)
else()
  target_compile_options(cusymint PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:>)
endif()

target_compile_features(cusymint PUBLIC cxx_std_14)
set_target_properties(cusymint PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                          CUDA_ARCHITECTURES OFF)

if(APPLE)
  set_property(TARGET cusymint PROPERTY BUILD_RPATH
                                        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()