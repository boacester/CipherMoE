cmake_minimum_required(VERSION 3.16.3)
project(CipherMoE LANGUAGES CXX)
find_package(OpenFHE 1.5.1 EXACT CONFIG REQUIRED)
add_library(experiment_options INTERFACE)
target_compile_features(experiment_options INTERFACE cxx_std_17)
target_include_directories(experiment_options INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(experiment_options SYSTEM INTERFACE
    "${OpenFHE_INCLUDE}" "${OpenFHE_INCLUDE}/third-party/include"
    "${OpenFHE_INCLUDE}/core" "${OpenFHE_INCLUDE}/pke" "${OpenFHE_INCLUDE}/binfhe")
separate_arguments(OPENFHE_COMPILE_OPTIONS UNIX_COMMAND "${OpenFHE_CXX_FLAGS}")
target_compile_options(experiment_options INTERFACE ${OPENFHE_COMPILE_OPTIONS})
target_link_libraries(experiment_options INTERFACE ${OpenFHE_SHARED_LIBRARIES})
add_executable(matrix_benchmark experiments/matrix_benchmark.cpp
    baseline/Dense/dense.cpp baseline/Select_Weight/select_weight.cpp
    baseline/Routing_Publicity/public_routing.cpp basis/basis_apply.cpp)
target_link_libraries(matrix_benchmark PRIVATE experiment_options)
add_executable(lut_benchmark basis/fused_lut.cpp)
target_link_libraries(lut_benchmark PRIVATE experiment_options)
