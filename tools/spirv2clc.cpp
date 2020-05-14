// Copyright 2020-2022 The spirv2clc authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fstream>
#include <iostream>
#include <string>

#include "spirv2clc.h"

void fail_help(const char *prog) {
  std::cerr << "Usage: " << prog << " [ --asm ] input.spv[asm]" << std::endl;
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

  bool input_asm = false;

  int arg = 1;

  if (argc < 2) {
    fail_help(argv[0]);
  }

  int num_options = 0;
  while (arg < argc) {
    if (!strcmp(argv[arg], "--asm")) {
      input_asm = true;
      num_options++;
    } else if (!strncmp(argv[arg], "--", 2)) {
      std::cerr << "Unknown option '" << argv[arg] << "'" << std::endl;
      fail_help(argv[0]);
    }
    arg++;
  }

  if (argc < 2 + num_options) {
    fail_help(argv[0]);
  }

  const char *fname = argv[argc - 1];
  std::ifstream file(fname, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Could not open " << fname << std::endl;
    exit(EXIT_FAILURE);
  }

  spirv2clc::translator translator;
  std::string srcgen;
  int err;
  if (input_asm) {
    std::stringstream buffer;
    buffer << file.rdbuf();

    err = translator.translate(buffer.str(), &srcgen);
  } else {
    file.seekg(0, std::ios::end);
    uint32_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> binary(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(binary.data()), size);
    err = translator.translate(binary, &srcgen);
  }

  printf("%s\n", srcgen.c_str());

  if (err != 0) {
    std::cerr << "Failed to translate module." << std::endl;
    exit(EXIT_FAILURE);
  }

  return 0;
}
