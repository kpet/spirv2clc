#include <layer.hpp>
#include <layer_surface.hpp>

#include "spirv2clc.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <string_view>
#include <vector>

std::optional<std::string> get_environment(const std::string &variable) {
#ifdef __ANDROID__
  // TODO: Implement get_environment for android
  return "";
#elif defined(_WIN32)
  size_t var_size;
  std::string temp;
  errno_t err = getenv_s(&var_size, NULL, 0, variable.c_str());
  if (var_size == 0 || err != 0)
    return std::nullopt;
  temp.resize(var_size);
  err = getenv_s(&var_size, temp.data(), var_size, variable.c_str());
  if (err == 0) {
    temp.pop_back(); // pop the null-terminator
    return std::move(temp);
  }
  return std::nullopt;
#else
  const char *output = std::getenv(variable.c_str());
  if (output != nullptr) {
    return std::string(output);
  } else
    return std::nullopt;
#endif
}

namespace spirv2clc {

cl_version translate_version_string(const std::string &ver) {
  // version strings take the form
  // OpenCL<space><major_version.minor_version><space><vendor-specific
  // information>
  auto major_start = ver.find(' ') + 1;
  auto major_end = ver.find('.', major_start);
  auto minor_start = major_end + 1;
  auto minor_end = ver.find(' ', minor_start);

  auto major =
      std::stoi(std::string(ver.data() + major_start, major_end - major_start));
  auto minor =
      std::stoi(std::string(ver.data() + minor_start, minor_end - minor_start));

  return CL_MAKE_VERSION(major, minor, 0);
}

spv_target_env translate_cl_version(cl_version ver) {
  switch (ver) {
  case CL_MAKE_VERSION(1, 2, 0):
    return SPV_ENV_OPENCL_1_2;
  case CL_MAKE_VERSION(2, 0, 0):
    return SPV_ENV_OPENCL_2_0;
  case CL_MAKE_VERSION(2, 1, 0):
    return SPV_ENV_OPENCL_2_1;
  case CL_MAKE_VERSION(2, 2, 0):
    return SPV_ENV_OPENCL_2_2;
  default:
    return SPV_ENV_OPENCL_1_2;
  }
}

std::string translate_spirv_target_env(spv_target_env env) {
  switch (env) {
  case SPV_ENV_OPENCL_1_2:
    return "SPV_ENV_OPENCL_1_2";
  case SPV_ENV_OPENCL_2_0:
    return "SPV_ENV_OPENCL_2_0";
  case SPV_ENV_OPENCL_2_1:
    return "SPV_ENV_OPENCL_2_1";
  case SPV_ENV_OPENCL_2_2:
    return "SPV_ENV_OPENCL_2_2";
  default:
    return "Unkown";
  }
}

layer::layer() noexcept : tracing{false} {
  init_dispatch();

  if (auto SPIRV2CLC_ENABLE_TRACE = get_environment("SPIRV2CLC_ENABLE_TRACE");
      SPIRV2CLC_ENABLE_TRACE) {
    tracing = std::atoi(SPIRV2CLC_ENABLE_TRACE.value().c_str());
  } else
    tracing = false;
}

void layer::init_dispatch() {
  dispatch.clGetPlatformInfo = &clGetPlatformInfo_wrap;
  dispatch.clGetDeviceInfo = &clGetDeviceInfo_wrap;
  dispatch.clCreateProgramWithIL = &clCreateProgramWithIL_wrap;
  dispatch.clBuildProgram = &clBuildProgram_wrap;
  dispatch.clGetProgramInfo = &clGetProgramInfo_wrap;
}

cl_icd_dispatch &layer::get_dispatch() { return dispatch; }

const cl_icd_dispatch *layer::get_target_dispatch() { return tdispatch; }

cl_int layer::clGetPlatformInfo(cl_platform_id platform,
                                cl_platform_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  switch (param_name) {
  case CL_PLATFORM_EXTENSIONS:
    return clGetPlatformInfo_CL_PLATFORM_EXTENSIONS(
        platform, param_name, param_value_size, param_value,
        param_value_size_ret);
    break;
  default:
    return tdispatch->clGetPlatformInfo(platform, param_name, param_value_size,
                                        param_value, param_value_size_ret);
    break;
  }
}

cl_int layer::clGetDeviceInfo(cl_device_id device, cl_device_info param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret) {
  switch (param_name) {
  case CL_DEVICE_EXTENSIONS:
    return clGetDeviceInfo_CL_DEVICE_EXTENSIONS(device, param_name,
                                                param_value_size, param_value,
                                                param_value_size_ret);
    break;
  case CL_DEVICE_IL_VERSION:
    return clGetDeviceInfo_CL_DEVICE_IL_VERSION(device, param_name,
                                                param_value_size, param_value,
                                                param_value_size_ret);
    break;
  case CL_DEVICE_ILS_WITH_VERSION:
    return clGetDeviceInfo_CL_DEVICE_ILS_WITH_VERSION(
        device, param_name, param_value_size, param_value,
        param_value_size_ret);
    break;
  default:
    return tdispatch->clGetDeviceInfo(device, param_name, param_value_size,
                                      param_value, param_value_size_ret);
    break;
  }
}

cl_program layer::clCreateProgramWithIL(cl_context context, const void *il,
                                        size_t length, cl_int *errcode_ret) {
  size_t devices_ret_size = 0;
  tdispatch->clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr,
                              &devices_ret_size);
  std::vector<cl_device_id> devices(devices_ret_size / sizeof(cl_device_id));
  tdispatch->clGetContextInfo(context, CL_CONTEXT_DEVICES,
                              devices.size() * sizeof(cl_device_id),
                              devices.data(), nullptr);

  std::vector<cl_version> versions;
  std::transform(devices.cbegin(), devices.cend(), std::back_inserter(versions),
                 [](const cl_device_id &device) {
                   size_t ret_size = 0;
                   tdispatch->clGetDeviceInfo(device, CL_DEVICE_VERSION, 0,
                                              nullptr, &ret_size);
                   std::string str(ret_size, '\0');
                   tdispatch->clGetDeviceInfo(device, CL_DEVICE_VERSION,
                                              ret_size, str.data(), nullptr);
                   return translate_version_string(str);
                 });

  auto min_ver = std::min_element(versions.begin(), versions.end());

  auto env = translate_cl_version(*min_ver);

  SPIRV2CLC_TRACE(
      "SPIR-V target env chosen on based on devices in context: %s\n",
      translate_spirv_target_env(env).c_str());

  spirv2clc::translator translator(env);
  std::string srcgen;
  std::vector<uint32_t> binary(static_cast<const uint32_t *>(il),
                               static_cast<const uint32_t *>(il) +
                                   length / sizeof(uint32_t));
  int err = translator.translate(binary, &srcgen);

  if (err == 0) {
    const char *str = srcgen.data();
    size_t len = srcgen.length();
    cl_program prog = tdispatch->clCreateProgramWithSource(context, 1, &str,
                                                           &len, errcode_ret);
    if (*errcode_ret == CL_SUCCESS) {
      program_ils[prog] =
          std::make_pair(std::string(static_cast<const char *>(il), length),
                         std::string(str, len));
    }
    return prog;
  } else {
    return tdispatch->clCreateProgramWithIL(context, il, length, errcode_ret);
  }
}

cl_int layer::clBuildProgram(cl_program program, cl_uint num_devices,
                             const cl_device_id *device_list,
                             const char *options,
                             void(CL_CALLBACK *pfn_notify)(cl_program program,
                                                           void *user_data),
                             void *user_data) {
  SPIRV2CLC_TRACE("Entering clBuildProgram\n");
  return tdispatch->clBuildProgram(program, num_devices, device_list, options,
                                   pfn_notify, user_data);
}

cl_int layer::clGetProgramInfo(cl_program program, cl_device_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  SPIRV2CLC_TRACE("Entering clGetProgramInfo\n");
  if (param_name == CL_PROGRAM_IL || param_name == CL_PROGRAM_IL_KHR)
    return clGetProgramInfo_CL_PROGRAM_IL(program, param_name, param_value_size,
                                          param_value, param_value_size_ret);
  else
    return tdispatch->clGetProgramInfo(program, param_name, param_value_size,
                                       param_value, param_value_size_ret);
}

cl_int layer::clGetPlatformInfo_CL_PLATFORM_EXTENSIONS(
    cl_platform_id platform, cl_platform_info, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  size_t dispatch_param_value_size_ret = 0;
  cl_int err_ =
      tdispatch->clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 0, nullptr,
                                   &dispatch_param_value_size_ret);
  if (err_ != CL_SUCCESS)
    return err_;

  std::vector<char> result(dispatch_param_value_size_ret, '\0');
  err_ = tdispatch->clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS,
                                      result.size(), result.data(), nullptr);
  if (err_ != CL_SUCCESS)
    return err_;

  std::string_view dispatch_extensions_string{result.data(), result.size()};
  static const char *spirv_ext_name = "cl_khr_il_program";
  if (dispatch_extensions_string.find(spirv_ext_name) == std::string::npos) {
    SPIRV2CLC_TRACE("Appending platform extension string, because "
                    "cl_khr_il_program was reported missing.\n");
    result.pop_back(); // pop the null-terminator
    result.push_back(' ');
    std::copy(spirv_ext_name, spirv_ext_name + std::strlen(spirv_ext_name),
              std::back_inserter(result));
    result.push_back('\0');
  } else
    SPIRV2CLC_TRACE("No need to alter platform extension string, because "
                    "cl_khr_il_program was reported present.\n");

  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  if (param_value_size_ret)
    *param_value_size_ret = result.size();

  if (param_value_size && param_value_size < result.size())
    return CL_INVALID_VALUE;

  if (param_value) {
    std::copy(result.begin(), result.end(), static_cast<char *>(param_value));
  }

  return CL_SUCCESS;
}

cl_int layer::clGetDeviceInfo_CL_DEVICE_EXTENSIONS(
    cl_device_id device, cl_device_info, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  cl_int err_ = CL_SUCCESS;
  size_t dispatch_param_value_size_ret = 0;
  err_ = tdispatch->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr,
                                    &dispatch_param_value_size_ret);
  if (err_ != CL_SUCCESS)
    return err_;

  std::vector<char> result(dispatch_param_value_size_ret, '\0');
  err_ = tdispatch->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, result.size(),
                                    result.data(), nullptr);
  if (err_ != CL_SUCCESS)
    return err_;

  std::string_view dispatch_extensions_string{result.data(), result.size()};
  static const char *spirv_ext_name = "cl_khr_il_program";
  if (dispatch_extensions_string.find(spirv_ext_name) == std::string::npos) {
    SPIRV2CLC_TRACE("Appending device extension string, because "
                    "cl_khr_il_program was reported missing.\n");
    result.pop_back(); // pop the null-terminator
    result.push_back(' ');
    std::copy(spirv_ext_name, spirv_ext_name + std::strlen(spirv_ext_name),
              std::back_inserter(result));
    result.push_back('\0');
  } else
    SPIRV2CLC_TRACE("No need to alter device extension string, because "
                    "cl_khr_il_program was reported present.\n");

  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  if (param_value_size_ret)
    *param_value_size_ret = result.size();

  if (param_value_size && param_value_size < result.size())
    return CL_INVALID_VALUE;

  if (param_value) {
    std::copy(result.begin(), result.end(), static_cast<char *>(param_value));
  }

  return CL_SUCCESS;
  /*
  cl_int err_;
  bool layer_has_nothing_to_do =
      device_supports_spirv_out_of_the_box(device, &err_);
  if (layer_has_nothing_to_do || err_ != CL_SUCCESS)
    return tdispatch->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS,
                                      param_value_size, param_value,
                                      param_value_size_ret);

  bool layer_has_work_but_need_not_alter_extension_string =
      spirv_queries_are_core_for_device(device, &err_);
  if (!layer_has_work_but_need_not_alter_extension_string || err_ != CL_SUCCESS)
    return tdispatch->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS,
                                      param_value_size, param_value,
                                      param_value_size_ret);

  size_t dispatch_param_value_size_ret = 0;
  err_ = tdispatch->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr,
                                    &dispatch_param_value_size_ret);
  if (err_ != CL_SUCCESS) return err_;

  std::string dispatch_extensions(dispatch_param_value_size_ret, '\0');
  err_ = tdispatch->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS,
                                    dispatch_extensions.size(),
                                    dispatch_extensions.data(), nullptr);
  if (err_ != CL_SUCCESS) return err_;
  dispatch_extensions.pop_back(); // pop null-terminator
  dispatch_extensions.append(" cl_khr_il_program");
  dispatch_extensions.push_back('\0');

  if (param_value_size_ret != nullptr)
    *param_value_size_ret = dispatch_extensions.length() * sizeof(char);

  if (param_value_size >= dispatch_extensions.length() * sizeof(char)) {
    if (param_value != nullptr) {
      std::copy(dispatch_extensions.begin(), dispatch_extensions.end(),
                static_cast<char *>(param_value));
      return CL_SUCCESS;
    } else
      return CL_INVALID_VALUE;
  } else if (param_value == nullptr)
    return CL_SUCCESS;
  else
    return CL_INVALID_VALUE;
  */
}

cl_int layer::clGetDeviceInfo_CL_DEVICE_IL_VERSION(
    cl_device_id device, cl_device_info, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  size_t dispatch_param_value_size_ret = 0;
  cl_int err_ = tdispatch->clGetDeviceInfo(
      device, CL_DEVICE_IL_VERSION, 0, nullptr, &dispatch_param_value_size_ret);

  std::vector<char> result(dispatch_param_value_size_ret, '\0');
  err_ = tdispatch->clGetDeviceInfo(device, CL_DEVICE_IL_VERSION, result.size(),
                                    result.data(), nullptr);
  if (err_ != CL_SUCCESS)
    return err_;

  std::string_view dispatch_extensions_string{result.data(), result.size()};
  static const char *spirv_il_name = "SPIR-V_1.0";
  if (dispatch_extensions_string.find(spirv_il_name) == std::string::npos) {
    SPIRV2CLC_TRACE("Appending il version string, because SPIR-V_1.0 was "
                    "reported missing.\n");
    result.pop_back(); // pop the null-terminator
    result.push_back(' ');
    std::copy(spirv_il_name, spirv_il_name + std::strlen(spirv_il_name),
              std::back_inserter(result));
    result.push_back('\0');
  } else
    SPIRV2CLC_TRACE("No need to alter il version string, because SPIR-V_1.0 "
                    "was reported present.\n");

  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  if (param_value_size_ret)
    *param_value_size_ret = result.size();

  if (param_value_size && param_value_size < result.size())
    return CL_INVALID_VALUE;

  if (param_value) {
    std::copy(result.begin(), result.end(), static_cast<char *>(param_value));
  }

  return CL_SUCCESS;
}

cl_int layer::clGetDeviceInfo_CL_DEVICE_ILS_WITH_VERSION(
    cl_device_id device, cl_device_info, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  size_t dispatch_param_value_size_ret = 0;
  cl_int err_ =
      tdispatch->clGetDeviceInfo(device, CL_DEVICE_ILS_WITH_VERSION, 0, nullptr,
                                 &dispatch_param_value_size_ret);

  std::vector<cl_name_version> result(dispatch_param_value_size_ret /
                                      sizeof(cl_name_version));
  err_ = tdispatch->clGetDeviceInfo(device, CL_DEVICE_ILS_WITH_VERSION,
                                    result.size(), result.data(), nullptr);
  if (err_ != CL_SUCCESS)
    return err_;

  static const auto spirv_il_name =
      cl_name_version{cl_version{CL_MAKE_VERSION(1, 0, 0)}, "SPIR-V"};
  if (std::find_if(result.cbegin(), result.cend(),
                   [=](const cl_name_version &name_ver) {
                     return name_ver.name == spirv_il_name.name &&
                            name_ver.version == spirv_il_name.version;
                   }) == result.cend()) {
    SPIRV2CLC_TRACE("Appending il name_version list, because SPIR-V 1.0 was "
                    "reported missing.\n");
    result.push_back(spirv_il_name);
  } else
    SPIRV2CLC_TRACE("No need to alter il name_version list, because SPIR-V 1.0 "
                    "was reported present.\n");

  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  if (param_value_size_ret)
    *param_value_size_ret = result.size();

  if (param_value_size && param_value_size < result.size())
    return CL_INVALID_VALUE;

  if (param_value) {
    std::copy(reinterpret_cast<char *>(result.data()),
              reinterpret_cast<char *>(result.data()) +
                  result.size() * sizeof(cl_name_version),
              static_cast<char *>(param_value));
  }

  return CL_SUCCESS;
}

cl_int layer::clGetProgramInfo_CL_PROGRAM_IL(cl_program program,
                                             cl_program_info,
                                             size_t param_value_size,
                                             void *param_value,
                                             size_t *param_value_size_ret) {
  if (param_value_size_ret != nullptr)
    *param_value_size_ret = program_ils[program].first.length() * sizeof(char);

  if (param_value_size >= program_ils[program].first.length() * sizeof(char)) {
    if (param_value != nullptr) {
      std::copy(program_ils[program].first.begin(),
                program_ils[program].first.end(),
                static_cast<char *>(param_value));
      return CL_SUCCESS;
    } else
      return CL_INVALID_VALUE;
  } else
    return CL_INVALID_VALUE;
}

} // namespace spirv2clc
