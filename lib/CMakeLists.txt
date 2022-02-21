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

add_library(libspirv2clc translator.cpp)

target_include_directories(libspirv2clc PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR} # For the export header
    ${OPENCL_HEADERS_DIR}
    ${SPIRV_TOOLS_DIR}
    ${SPIRV_TOOLS_DIR}/source
    ${SPIRV_TOOLS_BINARY_DIR})

set_target_properties(libspirv2clc PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_link_libraries(libspirv2clc SPIRV-Tools-opt)

generate_export_header(libspirv2clc)
