#pragma once

#include <CL/cl_icd.h> // cl_icd_dispatch

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo_wrap(
    cl_platform_id platform, cl_platform_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs_wrap(cl_platform_id platform,
                                                    cl_device_type device_type,
                                                    cl_uint num_entries,
                                                    cl_device_id *devices,
                                                    cl_uint *num_devices);

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo_wrap(
    cl_device_id device, cl_device_info param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clRetainDevice_wrap(cl_device_id device);

CL_API_ENTRY cl_int CL_API_CALL clReleaseDevice_wrap(cl_device_id device);

// Loader hooks

CL_API_ENTRY void *CL_API_CALL clGetExtensionFunctionAddress(const char *name);

CL_API_ENTRY cl_int CL_API_CALL clIcdGetPlatformIDsKHR(
    cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);
