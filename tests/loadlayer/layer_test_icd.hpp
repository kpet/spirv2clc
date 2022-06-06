#pragma once

#if defined(_WIN32)
#define NOMINMAX
#endif

#include <CL/cl_icd.h> // cl_icd_dispatch

#include <map>    // std::map<std::string, void*> _extensions
#include <memory> // std::unique_ptr
#include <set>
#include <string>  // std::string
#include <utility> // std::make_pair

namespace spirv2clc {
struct icd_compatible {
  cl_icd_dispatch *dispatch;

  icd_compatible();
  icd_compatible(const icd_compatible &) = delete;
  icd_compatible(icd_compatible &&) = delete;
  ~icd_compatible() = default;
  icd_compatible &operator=(const icd_compatible &) = delete;
  icd_compatible &operator=(icd_compatible &&) = delete;
};
} // namespace spirv2clc

struct _cl_device_id : public spirv2clc::icd_compatible {
  cl_int dev_type;
  std::string profile;
  std::string name;
  std::string vendor;
  std::string extensions;

  _cl_device_id();
  _cl_device_id(const _cl_device_id &) = delete;
  _cl_device_id(_cl_device_id &&) = delete;
  ~_cl_device_id() = default;
  _cl_device_id &operator=(const _cl_device_id &) = delete;
  _cl_device_id &operator=(_cl_device_id &&) = delete;

  cl_int clGetDeviceInfo(cl_device_info param_name, size_t param_value_size,
                         void *param_value, size_t *param_value_size_ret);

  cl_int clRetainDevice();

  cl_int clReleaseDevice();
};

struct _cl_context : public spirv2clc::icd_compatible {
  _cl_context();
  _cl_context(const _cl_context &) = delete;
  _cl_context(_cl_context &&) = delete;
  ~_cl_context() = default;
  _cl_context &operator=(const _cl_context &) = delete;
  _cl_context &operator=(_cl_context &&) = delete;
};

struct _cl_platform_id : public spirv2clc::icd_compatible {
  bool support_spirv;
  cl_version numeric_version;
  std::string profile;
  std::string version;
  std::string name;
  std::string vendor;
  std::string extensions;
  std::string suffix;

  _cl_platform_id();
  _cl_platform_id(const _cl_platform_id &) = delete;
  _cl_platform_id(_cl_platform_id &&) = delete;
  ~_cl_platform_id() = default;
  _cl_platform_id &operator=(const _cl_platform_id &) = delete;
  _cl_platform_id &operator=(_cl_platform_id &&) = delete;

  void init_dispatch();

  cl_int clGetPlatformInfo(cl_platform_info param_name, size_t param_value_size,
                           void *param_value, size_t *param_value_size_ret);

  cl_int clGetDeviceIDs(cl_device_type device_type, cl_uint num_entries,
                        cl_device_id *devices, cl_uint *num_devices);
};

namespace spirv2clc {
inline std::map<std::string, void *> _extensions{
    std::make_pair("clIcdGetPlatformIDsKHR",
                   reinterpret_cast<void *>(clIcdGetPlatformIDsKHR))};
inline cl_icd_dispatch _dispatch;
inline _cl_platform_id _platform;
inline std::set<std::shared_ptr<_cl_device_id>> _devices{
    std::make_shared<_cl_device_id>()};
inline std::set<std::shared_ptr<_cl_context>> _contexts;

template <typename T> cl_int CL_INVALID;
template <> inline cl_int CL_INVALID<cl_platform_id> = CL_INVALID_PLATFORM;
template <> inline cl_int CL_INVALID<cl_device_id> = CL_INVALID_DEVICE;
template <> inline cl_int CL_INVALID<cl_context> = CL_INVALID_CONTEXT;

template <typename T> T &_objects;
template <>
inline std::set<std::shared_ptr<_cl_device_id>> &_objects<cl_device_id> =
    _devices;
template <>
inline std::set<std::shared_ptr<_cl_context>> &_objects<cl_context> = _contexts;
} // namespace spirv2clc
