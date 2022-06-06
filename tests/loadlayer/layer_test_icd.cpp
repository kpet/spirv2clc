#include "layer_test_icd.hpp"
#include "layer_test_icd_surface.hpp"

#include <algorithm>
#include <iterator>
#include <optional>
#include <string>
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
icd_compatible::icd_compatible() : dispatch{&_dispatch} {}
} // namespace spirv2clc

_cl_device_id::_cl_device_id()
    : icd_compatible{}, dev_type{CL_DEVICE_TYPE_CUSTOM},
      profile{"FULL_PROFILE"}, name{"Test Device"}, vendor{"SPIRV2CLC Authors"},
      extensions{""} {
  extensions += "cl_khr_global_int32_base_atomics ";
  extensions += "cl_khr_global_int32_extended_atomics ";
  extensions += "cl_khr_local_int32_base_atomics ";
  extensions += "cl_khr_local_int32_extended_atomics";
  if (spirv2clc::_platform.support_spirv)
    extensions += "cl_khr_il_program";
}

cl_int _cl_device_id::clGetDeviceInfo(cl_device_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret) {
  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  std::vector<char> result;
  switch (param_name) {
  case CL_DEVICE_TYPE: {
    std::copy(reinterpret_cast<char *>(&dev_type),
              reinterpret_cast<char *>(&dev_type) + sizeof(dev_type),
              std::back_inserter(result));
    break;
  }
  case CL_DEVICE_NAME:
    std::copy(name.begin(), name.end(), std::back_inserter(result));
    result.push_back('\0');
    break;
  case CL_DEVICE_VENDOR:
    std::copy(vendor.begin(), vendor.end(), std::back_inserter(result));
    result.push_back('\0');
    break;
  case CL_DEVICE_PROFILE:
    std::copy(profile.begin(), profile.end(), std::back_inserter(result));
    result.push_back('\0');
    break;
  case CL_DEVICE_EXTENSIONS:
    std::copy(extensions.begin(), extensions.end(), std::back_inserter(result));
    result.push_back('\0');
    break;
  default:
    return CL_INVALID_VALUE;
  }

  if (param_value_size_ret)
    *param_value_size_ret = result.size();

  if (param_value_size && param_value_size < result.size())
    return CL_INVALID_VALUE;

  if (param_value) {
    std::copy(result.begin(), result.end(), static_cast<char *>(param_value));
  }

  return CL_SUCCESS;
}

_cl_platform_id::_cl_platform_id()
    : icd_compatible{}, numeric_version{}, version{}, vendor{"spirv2clc"},
      profile{"FULL_PROFILE"}, name{"SPIRV2CLC Layer Test ICD"},
      extensions{"cl_khr_icd"}, suffix{"spv2clc"} {
  init_dispatch();

  if (auto ICD_VERSION = get_environment("SPIRV2CLC_ICD_VERSION");
      ICD_VERSION) {
    auto icd_version = std::atoi(ICD_VERSION.value().c_str());
    numeric_version = CL_MAKE_VERSION(icd_version / 100, icd_version % 100 / 10,
                                      icd_version % 10);
  } else
    numeric_version = CL_MAKE_VERSION(1, 2, 0);

  version = std::string{"OpenCL "} +
            std::to_string(CL_VERSION_MAJOR(numeric_version)) + "." +
            std::to_string(CL_VERSION_MINOR(numeric_version)) + " Mock";

  if (auto ICD_SUPPORT_SPIRV = get_environment("SPIRV2CLC_ICD_SUPPORT_SPIRV");
      ICD_SUPPORT_SPIRV) {
    support_spirv = std::atoi(ICD_SUPPORT_SPIRV.value().c_str());
  } else
    support_spirv = false;

  if (support_spirv)
    extensions.append(" cl_khr_il_program");
}

void _cl_platform_id::init_dispatch() {
  dispatch->clGetPlatformInfo = clGetPlatformInfo_wrap;
  dispatch->clGetDeviceIDs = clGetDeviceIDs_wrap;
  dispatch->clGetDeviceInfo = clGetDeviceInfo_wrap;
  dispatch->clRetainDevice = clRetainDevice_wrap;
  dispatch->clReleaseDevice = clReleaseDevice_wrap;
}

cl_int _cl_platform_id::clGetPlatformInfo(cl_platform_info param_name,
                                          size_t param_value_size,
                                          void *param_value,
                                          size_t *param_value_size_ret) {
  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  std::string_view result;
  switch (param_name) {
  case CL_PLATFORM_PROFILE:
    result = profile;
    break;
  case CL_PLATFORM_VERSION:
    result = version;
    break;
  case CL_PLATFORM_NAME:
    result = name;
    break;
  case CL_PLATFORM_VENDOR:
    result = vendor;
    break;
  case CL_PLATFORM_EXTENSIONS:
    result = extensions;
    break;
  case CL_PLATFORM_ICD_SUFFIX_KHR:
    result = suffix;
    break;
  default:
    return CL_INVALID_VALUE;
  }

  if (param_value_size_ret)
    *param_value_size_ret = result.length() + 1;

  if (param_value_size && param_value_size < result.length() + 1)
    return CL_INVALID_VALUE;

  if (param_value) {
    std::copy(result.begin(), result.end(), static_cast<char *>(param_value));
    static_cast<char *>(param_value)[result.length()] = '\0';
  }

  return CL_SUCCESS;
}

cl_int _cl_platform_id::clGetDeviceIDs(cl_device_type device_type,
                                       cl_uint num_entries,
                                       cl_device_id *devices,
                                       cl_uint *num_devices) {
  using namespace spirv2clc;

  if (num_entries == 0 && devices != NULL)
    return CL_INVALID_VALUE;

  const bool asking_for_custom = device_type == CL_DEVICE_TYPE_CUSTOM ||
                                 device_type == CL_DEVICE_TYPE_DEFAULT ||
                                 device_type == CL_DEVICE_TYPE_ALL;

  if (num_devices)
    if (asking_for_custom)
      *num_devices = static_cast<cl_uint>(_devices.size());
    else
      *num_devices = 0;

  if (devices && asking_for_custom) {
    std::vector<std::shared_ptr<_cl_device_id>> result(_devices.cbegin(),
                                                       _devices.cend());

    std::transform(
        result.cbegin(),
        result.cbegin() + std::min<size_t>(num_entries, result.size()), devices,
        [](const std::shared_ptr<_cl_device_id> &dev) { return dev.get(); });
  }

  return CL_SUCCESS;
}

cl_int _cl_device_id::clRetainDevice() { return CL_SUCCESS; }

cl_int _cl_device_id::clReleaseDevice() { return CL_SUCCESS; }
