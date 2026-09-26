cmake_minimum_required(VERSION 3.16.3)
project(openfhe_check LANGUAGES CXX)

find_package(OpenFHE 1.5.1 EXACT CONFIG REQUIRED)

add_executable(openfhe-check main.cpp)
target_compile_features(openfhe-check PRIVATE cxx_std_17)
target_include_directories(openfhe-check SYSTEM PRIVATE
    "${OpenFHE_INCLUDE}"
    "${OpenFHE_INCLUDE}/third-party/include"
    "${OpenFHE_INCLUDE}/core"
    "${OpenFHE_INCLUDE}/pke"
    "${OpenFHE_INCLUDE}/binfhe")
separate_arguments(OPENFHE_COMPILE_OPTIONS UNIX_COMMAND "${OpenFHE_CXX_FLAGS}")
target_compile_options(openfhe-check PRIVATE ${OPENFHE_COMPILE_OPTIONS})
target_link_libraries(openfhe-check PRIVATE ${OpenFHE_SHARED_LIBRARIES})
