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

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>

#ifdef __APPLE__
#include <unistd.h>
#endif

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

static const std::string CLANG{"clang"};
static const std::string LLVMSPIRV{"llvm-spirv"};
static const std::string SPIRV2CLC{"spirv2clc"};

static decltype(&clCreateProgramWithSource) fnCreateProgramWithSource;
static decltype(&clRetainProgram) fnRetainProgram;
static decltype(&clReleaseProgram) fnReleaseProgram;
static decltype(&clBuildProgram) fnBuildProgram;
static decltype(&clCompileProgram) fnCompileProgram;
static decltype(&clLinkProgram) fnLinkProgram;
static decltype(&clGetProgramInfo) fnGetProgramInfo;
static decltype(&clGetProgramBuildInfo) fnGetProgramBuildInfo;
static decltype(&clCreateKernel) fnCreateKernel;
static decltype(&clCreateKernelsInProgram) fnCreateKernelsInProgram;
static decltype(&clGetKernelInfo) fnGetKernelInfo;

static struct init {
  init() {
#define LOAD_SYM(X) reinterpret_cast<decltype(&X)>(dlsym(RTLD_NEXT, #X))
    fnCreateProgramWithSource = LOAD_SYM(clCreateProgramWithSource);
    fnRetainProgram = LOAD_SYM(clRetainProgram);
    fnReleaseProgram = LOAD_SYM(clReleaseProgram);
    fnBuildProgram = LOAD_SYM(clBuildProgram);
    fnCompileProgram = LOAD_SYM(clCompileProgram);
    fnLinkProgram = LOAD_SYM(clLinkProgram);
    fnGetProgramInfo = LOAD_SYM(clGetProgramInfo);
    fnGetProgramBuildInfo = LOAD_SYM(clGetProgramBuildInfo);
    fnCreateKernel = LOAD_SYM(clCreateKernel);
    fnCreateKernelsInProgram = LOAD_SYM(clCreateKernelsInProgram);
    fnGetKernelInfo = LOAD_SYM(clGetKernelInfo);
#undef LOAD_SYM
  }
} gInit;

struct mock_program {
  mock_program(cl_context context, const std::string &&src)
      : m_context(context), m_src(src), m_build_status(CL_BUILD_NONE) {
    clRetainContext(m_context);
  }

  ~mock_program() { clReleaseContext(m_context); }

  cl_context context() const { return m_context; }
  const std::string &src() const { return m_src; }
  cl_build_status build_status() const { return m_build_status; }
  void set_build_status(cl_build_status status) { m_build_status = status; }
  const std::string &build_log() const { return m_build_log; }

private:
  cl_context m_context;
  std::string m_src;
  cl_build_status m_build_status;
  std::string m_build_log;
};

static std::unordered_map<mock_program *, cl_program> gProgramMap;

cl_program CL_API_CALL clCreateProgramWithSource(cl_context context,
                                                 cl_uint count,
                                                 const char **strings,
                                                 const size_t *lengths,
                                                 cl_int *errcode_ret) {
  // Get full source
  std::string src;
  for (cl_uint i = 0; i < count; i++) {
    if (lengths == nullptr) {
      src += strings[i];
    } else if (lengths[i] == 0) {
      src += strings[i];
    } else {
      src.append(strings[i], lengths[i]);
    }
  }

  auto prog = new mock_program(context, std::move(src));
  gProgramMap[prog] = nullptr;
  if (errcode_ret != nullptr) {
    *errcode_ret = CL_SUCCESS;
  }

  return reinterpret_cast<cl_program>(prog);
}

cl_int CL_API_CALL clRetainProgram(cl_program program) {
  auto prog = reinterpret_cast<mock_program *>(program);
  if (gProgramMap.count(prog)) {
    program = gProgramMap.at(prog);
    if (program == nullptr) {
      return CL_SUCCESS;
    }
  }
  return fnRetainProgram(program);
}

cl_int CL_API_CALL clReleaseProgram(cl_program program) {
  auto prog = reinterpret_cast<mock_program *>(program);
  if (gProgramMap.count(prog)) {
    program = gProgramMap.at(prog);
    if (program == nullptr) {
      return CL_SUCCESS;
    }
  }
  return fnReleaseProgram(program);
}

static bool save_string_to_file(const std::string &fname,
                                const std::string &text) {
  std::ofstream ofile{fname};

  if (!ofile.is_open()) {
    return false;
  }

  ofile.write(text.c_str(), text.size());
  ofile.close();

  return ofile.good();
}

std::ostream &log() {
  std::cout << "[SPIR2CL] ";
  return std::cout;
}

static cl_program compile(mock_program *program, cl_uint num_devices,
                          const cl_device_id *device_list, const char *options,
                          cl_uint num_input_headers = 0,
                          const cl_program *input_headers = nullptr,
                          const char **header_include_names = nullptr) {
  auto clprog = reinterpret_cast<cl_program>(program);
  // Store source to temporary file
  char tmp_template[] = "spirv2clc-XXXXXX";
  const char *tmp = mkdtemp(tmp_template);
  if (tmp == nullptr) {
    return nullptr;
  }

  std::filesystem::path tmp_folder{tmp};
  log() << "Created folder " << tmp_folder.string() << std::endl;

  std::filesystem::path src_file{tmp_folder / "original.cl"};
  if (!save_string_to_file(src_file, program->src())) {
    return nullptr;
  }

  for (unsigned i = 0; i < num_input_headers; i++) {
    std::filesystem::path header_file{tmp_folder / header_include_names[i]};
    std::filesystem::create_directories(header_file.parent_path());
    auto hprog = reinterpret_cast<mock_program *>(input_headers[i]);
    if (!save_string_to_file(header_file.string(), hprog->src())) {
      return nullptr;
    }
  }

  int syserr;

  // Translate to SPIR-V
  std::filesystem::path bitcode_file{tmp_folder / "original.bc"};
  std::string soptions;
  if (options != nullptr) {
    soptions += options;
  }

  cl_int err;

  // Select device
  // TODO support multiple devices
  if (num_devices > 1) {
    return nullptr;
  }
  cl_device_id device;
  if (device_list != nullptr) {
    device = device_list[0];
  } else {
    cl_uint num_devices;
    err = clGetProgramInfo(clprog, CL_PROGRAM_NUM_DEVICES, sizeof(num_devices),
                           &num_devices, nullptr);
    if (err != CL_SUCCESS) {
      return nullptr;
    }
    printf("num_devices = %u\n", num_devices);

    std::vector<cl_device_id> devices;
    devices.resize(num_devices);

    err = clGetProgramInfo(clprog, CL_PROGRAM_DEVICES,
                           devices.size() * sizeof(cl_device_id),
                           devices.data(), nullptr);
    if (err != CL_SUCCESS) {
      return nullptr;
    }

    device = devices[0];
  }

  // Choose target from device address size
  cl_uint address_bits;
  err = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(address_bits),
                        &address_bits, nullptr);
  if (err != CL_SUCCESS) {
    return nullptr;
  }
  std::string llvm_target;
  switch (address_bits) {
  case 32:
    llvm_target = "spir";
    break;
  case 64:
    llvm_target = "spir64";
    break;
  default:
    return nullptr;
  }
  // Choose CL C version from device query if not specified
  if (soptions.find("-cl-std=") == std::string::npos) {
    // TODO pick version from query
    soptions += " -cl-std=CL1.2 ";
  }
  std::string cmd_c_to_ir;
  cmd_c_to_ir += CLANG;
  cmd_c_to_ir += " -O0 -w -c ";
  cmd_c_to_ir += " -target " + llvm_target;
  cmd_c_to_ir += " -Xclang -no-opaque-pointers ";
  cmd_c_to_ir +=
      " -x cl " + soptions + " -Xclang -finclude-default-header -emit-llvm";
  cmd_c_to_ir += " -o " + bitcode_file.string();
  cmd_c_to_ir += " " + src_file.string();
  syserr = std::system(cmd_c_to_ir.c_str());
  if (syserr != 0) {
    log() << "Failed to compile OpenCL C to IR" << std::endl;
    return nullptr;
  }

  std::filesystem::path spv_file{tmp_folder / "original.spv"};
  std::string cmd_ir_to_spv = LLVMSPIRV;
  cmd_ir_to_spv += " --spirv-max-version=1.0";
  cmd_ir_to_spv += " -o " + spv_file.string();
  cmd_ir_to_spv += " " + bitcode_file.string();
  syserr = std::system(cmd_ir_to_spv.c_str());
  if (syserr != 0) {
    log() << "Failed to translate IR to SPIR-V" << std::endl;
    return nullptr;
  }

  // Translate SPIR-V back to C
  std::filesystem::path translated_source{tmp_folder / "translated.cl"};
  std::string cmd_spv_to_c =
      SPIRV2CLC + " " + spv_file.string() + " > " + translated_source.string();
  syserr = std::system(cmd_spv_to_c.c_str());
  if (syserr != 0) {
    log() << "Failed to translate SPIR-V to OpenCL C" << std::endl;
    return nullptr;
  }

  // Read back translated source
  std::string translated;
  std::ifstream ftranslated{translated_source};

  if (!ftranslated.is_open()) {
    std::cerr << "Could not open " << translated_source << std::endl;
    return nullptr;
  }

  std::stringstream buffer;
  buffer << ftranslated.rdbuf();
  translated = buffer.str();

  // Create programs
  const char *csrc = translated.c_str();
  clprog =
      fnCreateProgramWithSource(program->context(), 1, &csrc, nullptr, &err);
  if (err != CL_SUCCESS) {
    return nullptr;
  }
  gProgramMap[program] = clprog;

  for (unsigned i = 0; i < num_input_headers; i++) {
    auto hprog = reinterpret_cast<mock_program *>(input_headers[i]);
    const char *src = hprog->src().c_str();
    clprog =
        fnCreateProgramWithSource(program->context(), 1, &src, nullptr, &err);
    if (err != CL_SUCCESS) {
      return nullptr;
    }
    gProgramMap[hprog] = clprog;
  }

  return clprog;
}

cl_int CL_API_CALL clBuildProgram(
    cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
    void *user_data) {

  cl_int ret = CL_BUILD_SUCCESS;
  auto prog = reinterpret_cast<mock_program *>(program);
  if (gProgramMap.count(prog)) {
    // FIXME capture build log and store in wrapper
    auto program_sub = compile(prog, num_devices, device_list, options);
    if (program_sub == nullptr) {
      prog->set_build_status(CL_BUILD_ERROR);
      ret = CL_BUILD_PROGRAM_FAILURE;
    } else {
      program = program_sub;
    }
  }

  if (ret == CL_BUILD_SUCCESS) {
    ret = fnBuildProgram(program, num_devices, device_list, options, nullptr,
                         nullptr);
  }

  if (pfn_notify != nullptr) {
    pfn_notify(program, user_data);
  }

  return ret;
}

cl_int clCompileProgram(
    cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options, cl_uint num_input_headers,
    const cl_program *input_headers, const char **header_include_names,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
    void *user_data) {
  cl_int ret = CL_BUILD_SUCCESS;
  auto prog = reinterpret_cast<mock_program *>(program);
  if (gProgramMap.count(prog)) {
    // FIXME capture build log and store in wrapper
    auto program_sub =
        compile(prog, num_devices, device_list, options, num_input_headers,
                input_headers, header_include_names);
    if (program_sub == nullptr) {
      prog->set_build_status(CL_BUILD_ERROR);
      ret = CL_BUILD_PROGRAM_FAILURE;
    } else {
      program = program_sub;
    }
  }

  if (ret == CL_BUILD_SUCCESS) {
    std::vector<cl_program> header_programs;
    for (unsigned i = 0; i < num_input_headers; i++) {
      auto hprog = reinterpret_cast<mock_program *>(input_headers[i]);
      header_programs.push_back(gProgramMap.at(hprog));
    }

    ret = fnCompileProgram(program, num_devices, device_list, options,
                           num_input_headers, header_programs.data(),
                           header_include_names, nullptr, nullptr);
  }

  if (pfn_notify != nullptr) {
    pfn_notify(program, user_data);
  }

  return ret;
}

cl_program clLinkProgram(cl_context context, cl_uint num_devices,
                         const cl_device_id *device_list, const char *options,
                         cl_uint num_input_programs,
                         const cl_program *input_programs,
                         void(CL_CALLBACK *pfn_notify)(cl_program program,
                                                       void *user_data),
                         void *user_data, cl_int *errcode_ret) {

  // Build list of input programs
  std::vector<cl_program> input_progs;
  for (unsigned i = 0; i < num_input_programs; i++) {
    auto mock_input = reinterpret_cast<mock_program *>(input_programs[i]);
    if (gProgramMap.count(mock_input) == 0) {
      return nullptr;
    }
    auto p = gProgramMap.at(mock_input);
    input_progs.push_back(p);
  }
  auto ret = fnLinkProgram(context, num_devices, device_list, options,
                           num_input_programs, input_progs.data(), pfn_notify,
                           user_data, errcode_ret);

  return ret;
}

cl_int CL_API_CALL clGetProgramInfo(cl_program program,
                                    cl_program_info param_name,
                                    size_t param_value_size, void *param_value,
                                    size_t *param_value_size_ret) {

  auto prog = reinterpret_cast<mock_program *>(program);
  if (gProgramMap.count(prog)) {
    program = gProgramMap.at(prog);
    if (program == nullptr) {
      cl_int ret = CL_SUCCESS;
      const void *copy_ptr;
      size_t size_ret;
      cl_uint val_uint;
      std::vector<cl_device_id> val_devices;
      cl_context val_context;

      switch (param_name) {
      case CL_PROGRAM_NUM_DEVICES: {
        ret = clGetContextInfo(prog->context(), CL_CONTEXT_NUM_DEVICES,
                               sizeof(val_uint), &val_uint, nullptr);
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
      }
      case CL_PROGRAM_DEVICES: {
        ret = clGetContextInfo(prog->context(), CL_CONTEXT_NUM_DEVICES,
                               sizeof(val_uint), &val_uint, nullptr);
        if (ret != CL_SUCCESS) {
          break;
        }
        val_devices.resize(val_uint);
        ret = clGetContextInfo(prog->context(), CL_CONTEXT_DEVICES,
                               val_devices.size() * sizeof(cl_device_id),
                               val_devices.data(), nullptr);
        copy_ptr = val_devices.data();
        size_ret = val_devices.size() * sizeof(cl_device_id);
        break;
      }
      case CL_PROGRAM_SOURCE: {
        copy_ptr = prog->src().data();
        size_ret = prog->src().size() + 1; // Include NUL terminator
        break;
      }
      case CL_PROGRAM_CONTEXT: {
        val_context = prog->context();
        copy_ptr = &val_context;
        size_ret = sizeof(val_context);
        break;
      }
      case CL_PROGRAM_REFERENCE_COUNT: {
        val_uint = 1;
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
      }
      default:
        ret = CL_INVALID_VALUE;
      }

      if ((param_value != nullptr) && (param_value_size >= size_ret)) {
        memcpy(param_value, copy_ptr, size_ret);
      }

      if (param_value_size_ret != nullptr) {
        *param_value_size_ret = size_ret;
      }

      return ret;
    }
  }

  return fnGetProgramInfo(program, param_name, param_value_size, param_value,
                          param_value_size_ret);
}

cl_int CL_API_CALL clGetProgramBuildInfo(
    cl_program program, cl_device_id device, cl_program_build_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  auto prog = reinterpret_cast<mock_program *>(program);
  if (gProgramMap.count(prog)) {
    program = gProgramMap.at(prog);
    if (program == nullptr) {
      cl_int ret = CL_SUCCESS;
      const void *copy_ptr;
      size_t size_ret;
      cl_build_status val_build_status;

      switch (param_name) {
      case CL_PROGRAM_BUILD_STATUS:
        val_build_status = prog->build_status();
        copy_ptr = &val_build_status;
        size_ret = sizeof(val_build_status);
        break;
      case CL_PROGRAM_BUILD_LOG:
        copy_ptr = prog->build_log().data();
        size_ret = prog->build_log().size() + 1;
        break;
      default:
        ret = CL_INVALID_VALUE;
      }

      if ((param_value != nullptr) && (param_value_size >= size_ret)) {
        memcpy(param_value, copy_ptr, size_ret);
      }

      if (param_value_size_ret != nullptr) {
        *param_value_size_ret = size_ret;
      }

      return ret;
    }
  }

  return fnGetProgramBuildInfo(program, device, param_name, param_value_size,
                               param_value, param_value_size_ret);
}

cl_kernel CL_API_CALL clCreateKernel(cl_program program,
                                     const char *kernel_name,
                                     cl_int *errcode_ret) {
  auto prog = reinterpret_cast<mock_program *>(program);

  if (gProgramMap.count(prog)) {
    program = gProgramMap.at(prog);
  }

  return fnCreateKernel(program, kernel_name, errcode_ret);
}

cl_int CL_API_CALL clCreateKernelsInProgram(cl_program program,
                                            cl_uint num_kernels,
                                            cl_kernel *kernels,
                                            cl_uint *num_kernels_ret) {
  auto prog = reinterpret_cast<mock_program *>(program);

  if (gProgramMap.count(prog)) {
    program = gProgramMap.at(prog);
  }

  return fnCreateKernelsInProgram(program, num_kernels, kernels,
                                  num_kernels_ret);
}

cl_int clGetKernelInfo(cl_kernel kernel, cl_kernel_info param_name,
                       size_t param_value_size, void *param_value,
                       size_t *param_value_size_ret) {

  if ((param_name == CL_KERNEL_PROGRAM) && (param_value != nullptr) &&
      (param_value_size == sizeof(cl_program))) {
    cl_program kprog;
    cl_int err = fnGetKernelInfo(kernel, CL_KERNEL_PROGRAM, sizeof(kprog),
                                 &kprog, nullptr);

    if (err != CL_SUCCESS) {
      return err;
    }

    if (param_value_size_ret != nullptr) {
      *param_value_size_ret = sizeof(cl_program);
    }

    for (auto &mock_real : gProgramMap) {
      if (kprog == mock_real.second) {
        memcpy(param_value, &mock_real.first, sizeof(cl_program));
        return CL_SUCCESS;
      }
    }

    return CL_INVALID_KERNEL;
  }

  return fnGetKernelInfo(kernel, param_name, param_value_size, param_value,
                         param_value_size_ret);
}
