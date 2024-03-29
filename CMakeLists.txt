# Copyright 2020-2022 The spirv2clc authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.0)

project(spirv2clc)

cmake_policy(SET CMP0063 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GenerateExportHeader)

# Enable colour output wiht Ninja
include(CheckCXXCompilerFlag)

macro(AddCXXFlagIfSupported flag test)
   CHECK_CXX_COMPILER_FLAG(${flag} ${test})
   if( ${${test}} )
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
   endif()
endmacro()

if("Ninja" STREQUAL ${CMAKE_GENERATOR})
    AddCXXFlagIfSupported(-fdiagnostics-color COMPILER_SUPPORTS_COLOR_DIAGNOSTICS)
endif()

# Enable compiler warnings
if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# Add SPIRV-Tools
set(SPIRV_SKIP_TESTS ON)
set(SPIRV-Headers_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/SPIRV-Headers)
set(SPIRV_TOOLS_DIR ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools)
set(SPIRV_TOOLS_BINARY_DIR ${CMAKE_BINARY_DIR}/external/SPIRV-Tools)
add_subdirectory(${SPIRV_TOOLS_DIR} EXCLUDE_FROM_ALL)

set(OPENCL_HEADERS_DIR ${PROJECT_SOURCE_DIR}/external/OpenCL-Headers)

add_subdirectory(lib)
add_subdirectory(tools)

add_subdirectory(tests)
