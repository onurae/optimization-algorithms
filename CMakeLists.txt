# Minimum CMake version
cmake_minimum_required(VERSION 3.14)

# Name of the project
project(optimization-algorithms)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Google test
include(FetchContent)
message(STATUS "Fetching googletest library...")
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 58d77fa8070e8cec2dc1ed015d66b454c8d78850 # release-1.12.1
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing in CMake
enable_testing()

# Unit test project
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/golden-section/*.?pp)
file(GLOB TEST_SOURCES ${PROJECT_NAME}/golden-section/*.test.cpp)
add_executable(GoldenSection ${SOURCES} ${TEST_SOURCES})
target_link_libraries(GoldenSection
    PRIVATE GTest::gtest_main
)

# Unit test project
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/nelder-mead/*.?pp)
file(GLOB TEST_SOURCES ${PROJECT_NAME}/nelder-mead/*.test.cpp)
add_executable(NelderMead ${SOURCES} ${TEST_SOURCES})
target_link_libraries(NelderMead
    PRIVATE GTest::gtest_main
)

# Unit test project
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/particle-swarm/*.?pp)
file(GLOB TEST_SOURCES ${PROJECT_NAME}/particle-swarm/*.test.cpp)
add_executable(ParticleSwarm ${SOURCES} ${TEST_SOURCES})
target_link_libraries(ParticleSwarm
    PRIVATE GTest::gtest_main
)

# Enable CMake’s test runner to discover the tests
include(GoogleTest)
gtest_discover_tests(GoldenSection)
gtest_discover_tests(NelderMead)
gtest_discover_tests(ParticleSwarm)