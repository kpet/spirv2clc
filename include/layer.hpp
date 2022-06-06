#pragma once

#include <CL/cl_layer.h>

#include <map>
#include <memory>
#include <string>

namespace spirv2clc {
class layer {
public:
  layer() noexcept;
  layer(const layer &) = delete;
  layer(layer &&) = delete;
  ~layer() = default;
  layer &operator=(const layer &) = delete;
  layer &operator=(layer &&) = delete;

  cl_icd_dispatch &get_dispatch();
  const cl_icd_dispatch *get_target_dispatch();

  cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name,
                           size_t param_value_size, void *param_value,
                           size_t *param_value_size_ret);

  cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name,
                         size_t param_value_size, void *param_value,
                         size_t *param_value_size_ret);

  cl_program clCreateProgramWithIL(cl_context context, const void *il,
                                   size_t length, cl_int *errcode_ret);

  cl_int clBuildProgram(cl_program program, cl_uint num_devices,
                        const cl_device_id *device_list, const char *options,
                        void(CL_CALLBACK *pfn_notify)(cl_program program,
                                                      void *user_data),
                        void *user_data);

  cl_int clGetProgramInfo(cl_program program, cl_program_info param_name,
                          size_t param_value_size, void *param_value,
                          size_t *param_value_size_ret);

  std::map<cl_program, std::pair<std::string, std::string>> program_ils;

  bool tracing;

private:
  void init_dispatch(void);

  cl_int clGetPlatformInfo_CL_PLATFORM_EXTENSIONS(cl_platform_id platform,
                                                  cl_platform_info param_name,
                                                  size_t param_value_size,
                                                  void *param_value,
                                                  size_t *param_value_size_ret);

  cl_int clGetDeviceInfo_CL_DEVICE_EXTENSIONS(cl_device_id device,
                                              cl_device_info param_name,
                                              size_t param_value_size,
                                              void *param_value,
                                              size_t *param_value_size_ret);

  cl_int clGetDeviceInfo_CL_DEVICE_IL_VERSION(cl_device_id device,
                                              cl_device_info param_name,
                                              size_t param_value_size,
                                              void *param_value,
                                              size_t *param_value_size_ret);

  cl_int clGetDeviceInfo_CL_DEVICE_ILS_WITH_VERSION(
      cl_device_id device, cl_device_info param_name, size_t param_value_size,
      void *param_value, size_t *param_value_size_ret);

  cl_int clGetProgramInfo_CL_PROGRAM_IL(cl_program program,
                                        cl_program_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret);
};

inline layer instance;
inline const cl_icd_dispatch *tdispatch = nullptr;
inline cl_icd_dispatch dispatch{};

} // namespace spirv2clc

// internal tracing macros
#define SPIRV2CLC_TRACE(...)                                                   \
  do {                                                                         \
    if (spirv2clc::instance.tracing) {                                         \
      fprintf(stderr, "SPIRV2CLC trace at %s:%d: ", __FILE__, __LINE__);       \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#ifdef _WIN32
#define SPIRV2CLC_WIDE_TRACE(...)                                              \
  do {                                                                         \
    if (spirv2clc::instance.tracing) {                                         \
      fwprintf(stderr, L"SPIRV2CLC trace at %hs:%d: ", __FILE__, __LINE__);    \
      fwprintf(stderr, __VA_ARGS__);                                           \
    }                                                                          \
  } while (0)

#else
#define SPIRV2CLC_WIDE_TRACE(...)
#endif
