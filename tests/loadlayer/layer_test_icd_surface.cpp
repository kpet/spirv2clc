#include "layer_test_icd_surface.hpp"
#include "layer_test_icd.hpp"

#include <type_traits> // std::remove_pointer_t

template <typename T, typename F> cl_int invoke_if_valid(T cl_object, F &&f) {
  using namespace spirv2clc;

  auto it =
      std::find_if(_objects<T>.cbegin(), _objects<T>.cend(),
                   [=](const std::shared_ptr<std::remove_pointer_t<T>> &obj) {
                     return cl_object == obj.get();
                   });

  if (it != _objects<T>.cend())
    return f();
  else
    return CL_INVALID<T>;
}

template <typename F> cl_int invoke_if_valid(cl_platform_id platform, F &&f) {
  using namespace spirv2clc;

  if (platform == &_platform)
    return f();
  else
    return CL_INVALID<cl_platform_id>;
}

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo_wrap(
    cl_platform_id platform, cl_platform_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  return invoke_if_valid(platform, [&]() {
    return platform->clGetPlatformInfo(param_name, param_value_size,
                                       param_value, param_value_size_ret);
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs_wrap(cl_platform_id platform,
                                                    cl_device_type device_type,
                                                    cl_uint num_entries,
                                                    cl_device_id *devices,
                                                    cl_uint *num_devices) {
  return invoke_if_valid(platform, [&]() {
    return platform->clGetDeviceIDs(device_type, num_entries, devices,
                                    num_devices);
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo_wrap(
    cl_device_id device, cl_device_info param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  return invoke_if_valid(device, [&]() {
    return device->clGetDeviceInfo(param_name, param_value_size, param_value,
                                   param_value_size_ret);
  });
}

CL_API_ENTRY cl_int CL_API_CALL clRetainDevice_wrap(cl_device_id device) {
  return invoke_if_valid(device, [&]() { return device->clRetainDevice(); });
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseDevice_wrap(cl_device_id device) {
  return invoke_if_valid(device, [&]() { return device->clReleaseDevice(); });
}

// Loader hooks

CL_API_ENTRY void *CL_API_CALL clGetExtensionFunctionAddress(const char *name) {
  using namespace spirv2clc;

  auto it = _extensions.find(name);
  if (it != _extensions.end())
    return it->second;
  else
    return nullptr;
}

CL_API_ENTRY cl_int CL_API_CALL clIcdGetPlatformIDsKHR(
    cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
  using namespace spirv2clc;
  static constexpr cl_uint plat_count = 1;

  if (num_platforms)
    *num_platforms = plat_count;

  if ((platforms && num_entries > plat_count) ||
      (platforms && num_entries <= 0) || (!platforms && num_entries >= 1)) {
    return CL_INVALID_VALUE;
  }

  if (platforms && num_entries == plat_count)
    platforms[0] = &_platform;

  return CL_SUCCESS;
}
