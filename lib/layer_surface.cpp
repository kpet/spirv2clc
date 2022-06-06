#include <layer.hpp>
#include <layer_surface.hpp>

#if defined(_WIN32)
#define NOMINMAX
#endif

CL_API_ENTRY cl_int CL_API_CALL clGetLayerInfo(cl_layer_info param_name,
                                               size_t param_value_size,
                                               void *param_value,
                                               size_t *param_value_size_ret) {
  SPIRV2CLC_TRACE("Entering clGetLayerInfo\n");

  switch (param_name) {
  case CL_LAYER_API_VERSION:
    if (param_value) {
      if (param_value_size < sizeof(cl_layer_api_version))
        return CL_INVALID_VALUE;
      *((cl_layer_api_version *)param_value) = CL_LAYER_API_VERSION_100;
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(cl_layer_api_version);
    break;
  default:
    return CL_INVALID_VALUE;
  }
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clInitLayer(cl_uint num_entries, const cl_icd_dispatch *target_dispatch,
            cl_uint *num_entries_out,
            const struct _cl_icd_dispatch **layer_dispatch_ret) {
  SPIRV2CLC_TRACE("Entering clInitLayer\n");

  if (!target_dispatch || !layer_dispatch_ret || !num_entries_out ||
      num_entries <
          sizeof(_cl_icd_dispatch) / sizeof(_cl_icd_dispatch::clGetPlatformIDs))
    return CL_INVALID_VALUE;

  spirv2clc::tdispatch = target_dispatch;
  *layer_dispatch_ret = &spirv2clc::dispatch;
  *num_entries_out =
      sizeof(_cl_icd_dispatch) / sizeof(_cl_icd_dispatch::clGetPlatformIDs);

  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo_wrap(
    cl_platform_id platform, cl_platform_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  // SPIRV2CLC_TRACE("Entering clGetPlatformInfo_wrap\n");
  return spirv2clc::instance.clGetPlatformInfo(platform, param_name,
                                               param_value_size, param_value,
                                               param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo_wrap(
    cl_device_id device, cl_device_info param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  // SPIRV2CLC_TRACE("Entering clGetDeviceInfo_wrap\n");
  return spirv2clc::instance.clGetDeviceInfo(
      device, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithIL_wrap(
    cl_context context, const void *il, size_t length, cl_int *errcode_ret) {
  // SPIRV2CLC_TRACE("Entering clCreateProgramWithIL_wrap\n");
  return spirv2clc::instance.clCreateProgramWithIL(context, il, length,
                                                   errcode_ret);
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithILKHR_wrap(
    cl_context context, const void *il, size_t length, cl_int *errcode_ret) {
  // SPIRV2CLC_TRACE("Entering clCreateProgramWithILKHR_wrap\n");
  return spirv2clc::instance.clCreateProgramWithIL(context, il, length,
                                                   errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram_wrap(
    cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
    void *user_data) {
  // SPIRV2CLC_TRACE("Entering clBuildProgram_wrap\n");
  return spirv2clc::instance.clBuildProgram(program, num_devices, device_list,
                                            options, pfn_notify, user_data);
}

CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo_wrap(
    cl_program program, cl_program_info param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  // SPIRV2CLC_TRACE("Entering clGetProgramInfo_wrap\n");
  return spirv2clc::instance.clGetProgramInfo(
      program, param_name, param_value_size, param_value, param_value_size_ret);
}
