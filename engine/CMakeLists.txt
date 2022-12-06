cmake_minimum_required(VERSION 3.12)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cusymint LANGUAGES C CXX CUDA)

# Convenience variables

set(SRC_DIR "src")
set(UTILS_DIR "${SRC_DIR}/Utils")
set(SYMBOL_DIR "${SRC_DIR}/Symbol")
set(EVALUATION_DIR "${SRC_DIR}/Evaluation")
set(PARSER_DIR "${SRC_DIR}/Parser")
set(SERVER_DIR "${SRC_DIR}/Server")
set(SOLVER_DIR "${SRC_DIR}/Solver")
set(TEST_DIR "test")
set(TEST_PARSER_DIR "${TEST_DIR}/Parser")
set(TEST_SYMBOL_DIR "${TEST_DIR}/Symbol")
set(TEST_MATH_DIR "${TEST_DIR}/Math")
set(TEST_EVALUATION_DIR "${TEST_DIR}/Evaluation")

# Define build targets

add_library(
  libcusymint
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
  ${SYMBOL_DIR}/Hyperbolic.cu
  ${SYMBOL_DIR}/Polynomial.cu
  ${SYMBOL_DIR}/Logarithm.cu
  ${SYMBOL_DIR}/SubexpressionCandidate.cu
  ${SYMBOL_DIR}/SubexpressionVacancy.cu
  ${EVALUATION_DIR}/Collapser.cu
  ${EVALUATION_DIR}/Integrator.cu
  ${EVALUATION_DIR}/IntegratorKernels.cu
  ${EVALUATION_DIR}/StaticFunctions.cu
  ${EVALUATION_DIR}/Heuristic/Heuristic.cu
  ${EVALUATION_DIR}/Heuristic/SplitSum.cu
  ${EVALUATION_DIR}/Heuristic/SubstituteEToX.cu
  ${EVALUATION_DIR}/Heuristic/UniversalSubstitution.cu
  ${EVALUATION_DIR}/Heuristic/BringOutConstantsFromProduct.cu
  ${EVALUATION_DIR}/Heuristic/LinearSubstitution.cu
  ${EVALUATION_DIR}/KnownIntegral/KnownIntegral.cu
  ${EVALUATION_DIR}/KnownIntegral/ConstantIntegral.cu
  ${EVALUATION_DIR}/KnownIntegral/PowerFunction.cu
  ${EVALUATION_DIR}/KnownIntegral/SimpleSineCosine.cu
  ${EVALUATION_DIR}/KnownIntegral/SimpleExponent.cu
  ${EVALUATION_DIR}/KnownIntegral/SimpleVariable.cu
  ${EVALUATION_DIR}/KnownIntegral/SimpleArctan.cu
  ${PARSER_DIR}/Scanner.cu
  ${PARSER_DIR}/Parser.cu)

add_executable(cusymint ${SRC_DIR}/main.cu)

add_executable(
  testcusymint
  ${TEST_PARSER_DIR}/Scanner.cu
  ${TEST_PARSER_DIR}/Parser.cu
  ${TEST_DIR}/Heuristic.cu
  ${TEST_DIR}/IntegralCommons.cu
  ${TEST_DIR}/KnownIntegral.cu
  ${TEST_DIR}/Simplify.cu
  ${TEST_DIR}/Symbol.cu
  ${TEST_DIR}/Derivatives.cu
  ${TEST_SYMBOL_DIR}/MetaOperators.cu
  ${TEST_MATH_DIR}/PascalTriangle.cu
  ${TEST_EVALUATION_DIR}/IntegratorUtils.cu
  ${TEST_EVALUATION_DIR}/Kernels.cu
  ${TEST_EVALUATION_DIR}/Collapser.cu)

add_executable(
  srvcusymint
  ${SRC_DIR}/main_server.cu ${SERVER_DIR}/ResponseBuilder.cu
  ${SERVER_DIR}/Server.cu ${SOLVER_DIR}/CachedParser.cu
  ${SOLVER_DIR}/Expression.cu ${SOLVER_DIR}/Solver.cu
  ${SERVER_DIR}/SolverProcessManager.cu ${SERVER_DIR}/JsonSolver.cu
  ${SERVER_DIR}/Logger.cu)

# Debug build related things

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "DEBUG")
endif()

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)

if(CMAKE_BUILD_TYPE_UPPER MATCHES DEBUG)
  target_compile_options(libcusymint PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: -g -G >)
  add_compile_definitions(__DEBUG__)
else()
  target_compile_options(libcusymint PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:>)
endif()

# Compilation options

target_compile_features(libcusymint PUBLIC cxx_std_17)
set_target_properties(
  libcusymint testcusymint cusymint srvcusymint
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES OFF)

# Something to make this work on Mac OS

if(APPLE)
  set_target_properties(
    libcusymint
    cusymint
    testcusymint
    srvcusymint
    PROPERTY
    BUILD_RPATH
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

# Additional packages

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1)

set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)

FetchContent_Declare(
  mongoose
  GIT_REPOSITORY https://github.com/cesanta/mongoose.git
  GIT_TAG 7.8)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 9.1.0)
FetchContent_MakeAvailable(fmt)

FetchContent_MakeAvailable(googletest mongoose fmt)

add_library(libmongoose ${mongoose_SOURCE_DIR}/mongoose.c)

# Includes and linking

target_include_directories(libcusymint PUBLIC ${SRC_DIR} ${mongoose_SOURCE_DIR})
target_link_libraries(libcusymint PUBLIC fmt::fmt)

target_link_libraries(cusymint PRIVATE libcusymint)

target_link_libraries(srvcusymint PRIVATE libcusymint libmongoose)

target_link_libraries(testcusymint PRIVATE libcusymint gtest_main)
