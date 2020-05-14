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

add_library(testlayer SHARED testlayer.cpp)
target_include_directories(testlayer PRIVATE ${OPENCL_HEADERS_DIR})
target_link_libraries(testlayer dl)
