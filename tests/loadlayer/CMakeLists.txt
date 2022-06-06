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

if (APPLE)
set(CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
endif()

add_library(layerloadicd SHARED
  layer_test_icd.cpp
  layer_test_icd_surface.cpp
  $<$<CXX_COMPILER_ID:GNU>:layer_test_icd.map>
  $<$<AND:$<PLATFORM_ID:Windows>,$<CXX_COMPILER_ID:MSVC,Clang>>:layer_test_icd.def>
)
target_compile_features(layerloadicd PRIVATE cxx_std_17)
target_include_directories(layerloadicd PRIVATE ${OPENCL_HEADERS_DIR})
target_link_libraries(layerloadicd ${CMAKE_DL_LIBS})
target_compile_definitions(layerloadicd PRIVATE CL_TARGET_OPENCL_VERSION=300)

add_executable(layerloadtest layerloadtest.cpp)
find_package(OpenCL CONFIG REQUIRED COMPONENTS OpenCL HeadersCpp)
target_link_libraries(layerloadtest PRIVATE OpenCL::OpenCL OpenCL::HeadersCpp)
target_compile_definitions(layerloadtest
  PRIVATE
    CL_HPP_TARGET_OPENCL_VERSION=300
    CL_HPP_ENABLE_EXCEPTIONS
)

foreach(ICD_VERSION IN ITEMS 120 210 300)
  foreach(ICD_SUPPORT_SPIRV IN ITEMS 0 1)
  add_test(
    NAME layerloadtest_${ICD_VERSION}_${ICD_SUPPORT_SPIRV}
    COMMAND layerloadtest 0 0
  )

  set(TEST_ENVIRONMENT
    OCL_ICD_ENABLE_TRACE=1
    OCL_ICD_FILENAMES=$<TARGET_FILE:layerloadicd>
    OPENCL_LAYERS=$<TARGET_FILE:SPIRV2CLC-Layer>
    SPIRV2CLC_ICD_VERSION=${ICD_VERSION}
    SPIRV2CLC_ICD_SUPPORT_SPIRV=${ICD_SUPPORT_SPIRV}
    SPIRV2CLC_ENABLE_TRACE=1
  )

  set_tests_properties(layerloadtest_${ICD_VERSION}_${ICD_SUPPORT_SPIRV}
    PROPERTIES
      ENVIRONMENT "${TEST_ENVIRONMENT}"
      PASS_REGULAR_EXPRESSION "$<IF:$<BOOL:${ICD_SUPPORT_SPIRV}>,present,missing>"
  )
  endforeach()
endforeach()
