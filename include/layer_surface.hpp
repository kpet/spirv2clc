#pragma once

#include <CL/cl_layer.h>

CL_API_ENTRY cl_int CL_API_CALL clGetLayerInfo(cl_layer_info param_name,
                                               size_t param_value_size,
                                               void *param_value,
                                               size_t *param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clInitLayer(
    cl_uint num_entries, const _cl_icd_dispatch *target_dispatch,
    cl_uint *num_entries_out, const _cl_icd_dispatch **layer_dispatch_ret);

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo_wrap(
    cl_platform_id platform, cl_platform_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo_wrap(
    cl_device_id device, cl_device_info param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret);

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithIL_wrap(
    cl_context context, const void *il, size_t length, cl_int *errcode_ret);

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithILKHR_wrap(
    cl_context context, const void *il, size_t length, cl_int *errcode_ret);

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram_wrap(
    cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
    void *user_data);

CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo_wrap(
    cl_program program, cl_program_info param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret);
