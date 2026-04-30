cmake_minimum_required(VERSION 3.20)
project(neutronics VERSION 1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion -Wmissing-braces -Wnon-virtual-dtor")

# Set default build type to Release.
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_definitions(-DBUILD_TYPE_DEBUG)
endif()
 
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "deal.II dir: ${DEAL_II_DIR}")

# Locate Boost.
find_package(Boost 1.74.0 REQUIRED
  COMPONENTS filesystem iostreams serialization
  HINTS ${BOOST_DIR} $ENV{BOOST_DIR} $ENV{mkBoostPrefix})
message(STATUS "Using the Boost-${Boost_VERSION} configuration found at ${Boost_DIR}")
message(STATUS)
include_directories(${Boost_INCLUDE_DIRS})

# Locate deal.II and initialize its variables.
find_package(deal.II 9.3.1 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

# nlohmann/json
include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
 
FetchContent_MakeAvailable(nlohmann_json)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Files
add_executable(neutronics
  src/CrossSectionManager.cpp
  src/ZeroModeOperator.cpp
  src/SecondModeOperator.cpp
  src/CouplingOperator.cpp
  src/MultigridPreconditioner.cpp
  src/InverseOperator.cpp
  src/BlockGSPreconditioner.cpp
  src/BlockDiagonalPreconditioner.cpp
  src/SP3Operator.cpp
  src/EnergyGroup.cpp
  src/NeutronSolver.cpp
  src/main.cpp
)
deal_ii_setup_target(neutronics)
target_link_libraries(neutronics nlohmann_json::nlohmann_json)
target_include_directories(neutronics PRIVATE ${CMAKE_SOURCE_DIR}/include)