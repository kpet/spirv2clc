#include "spirv2clc.h"

#include <CL/cl_layer.h>

#include <vector>
#include <string_view>
#include <sstream>
#include <iterator>
#include <algorithm>

static struct _cl_icd_dispatch dispatch = {};

static const struct _cl_icd_dispatch *tdispatch;

  /* Layer API entry points */
CL_API_ENTRY cl_int CL_API_CALL
clGetLayerInfo(
    cl_layer_info  param_name,
    size_t         param_value_size,
    void          *param_value,
    size_t        *param_value_size_ret) {
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

static void _init_dispatch(void);

CL_API_ENTRY cl_int CL_API_CALL
clInitLayer(
    cl_uint                         num_entries,
    const struct _cl_icd_dispatch  *target_dispatch,
    cl_uint                        *num_entries_out,
    const struct _cl_icd_dispatch **layer_dispatch_ret) {
  if (!target_dispatch || !layer_dispatch_ret ||!num_entries_out || num_entries < sizeof(dispatch)/sizeof(dispatch.clGetPlatformIDs))
    return CL_INVALID_VALUE;

  tdispatch = target_dispatch;
  _init_dispatch();

  *layer_dispatch_ret = &dispatch;
  *num_entries_out = sizeof(dispatch)/sizeof(dispatch.clGetPlatformIDs);
  return CL_SUCCESS;
}

// Layer logic

inline cl_int clGetDeviceInfo_CL_DEVICE_EXTENSIONS(
  cl_device_id device,
  cl_device_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  // If querying extensions, check if SPIR-V is supported
  size_t dispatch_param_value_size_ret = 0;
  cl_int err_ = tdispatch->clGetDeviceInfo(
    device,
    CL_DEVICE_EXTENSIONS,
    0,
    nullptr,
    &dispatch_param_value_size_ret);
  if (err_ == CL_SUCCESS)
  {
    std::vector<char> dispatch_extensions(
      dispatch_param_value_size_ret,
      '\0'
    );
    err_ = tdispatch->clGetDeviceInfo(
      device,
      CL_DEVICE_EXTENSIONS,
      dispatch_extensions.size(),
      dispatch_extensions.data(),
      nullptr
    );
    std::string_view dispatch_extensions_string{
      dispatch_extensions.data(),
      dispatch_extensions.size()
    };
    #define spirv_ext_name "cl_khr_il_program"
    if (dispatch_extensions_string.find(spirv_ext_name) != std::string::npos)
    {
      // If SPIR-V is supported, pass along invocation unchanged
      return tdispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_EXTENSIONS,
        param_value_size,
        param_value,
        param_value_size_ret);
    }
    else
    {
      // If not supported, layer has to append to supported extensions
      size_t layer_param_value_size_ret =
        dispatch_param_value_size_ret +
        1 + // space
        sizeof(spirv_ext_name);
      if (param_value_size_ret != nullptr)
        *param_value_size_ret = layer_param_value_size_ret;
      std::stringstream layer_extensions_stream;
      layer_extensions_stream <<
        dispatch_extensions_string <<
        " " <<
        spirv_ext_name;
      if (param_value_size >= layer_param_value_size_ret)
      {
        if (param_value != nullptr)
        {
          std::copy(
            std::istreambuf_iterator<char>{layer_extensions_stream},
            std::istreambuf_iterator<char>{},
            static_cast<char*>(param_value)
          );
          return CL_SUCCESS;
        }
        else
          return CL_INVALID_VALUE;
      }
      else
        return CL_INVALID_VALUE;
    }
    #undef spirv_ext_name
  }
  else
    return err_;
}

inline cl_int clGetDeviceInfo_CL_DEVICE_IL_VERSION(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  // TODO: check platform version of device if the query is
  //       available to begin with
  return CL_SUCCESS;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo_wrap(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  switch (param_name)
  {
    case CL_DEVICE_EXTENSIONS:
      clGetDeviceInfo_CL_DEVICE_EXTENSIONS(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret
      );
      break;
    case CL_DEVICE_IL_VERSION:
      clGetDeviceInfo_CL_DEVICE_IL_VERSION(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret
      );
      break;
    default:
      return tdispatch->clGetDeviceInfo(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
      break;
  }
}

static CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithIL_wrap(
    cl_context context,
    const void* il,
    size_t length,
    cl_int* errcode_ret)
{
  cl_program program = tdispatch->clCreateProgramWithIL(
    context,
    il,
    length,
    errcode_ret);
  return program;
}

static CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithBinary_wrap(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const size_t* lengths,
    const unsigned char** binaries,
    cl_int* binary_status,
    cl_int* errcode_ret)
{
  spirv2clc::translator translator;
  std::string srcgen;
  std::vector<uint32_t> binary;

  int err = translator.translate(binary, &srcgen);
  cl_program program = tdispatch->clCreateProgramWithBinary(
    context,
    num_devices,
    device_list,
    lengths,
    binaries,
    binary_status,
    errcode_ret);
  return program;
}

static CL_API_ENTRY cl_int CL_API_CALL clBuildProgram_wrap(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
  return tdispatch->clBuildProgram(
            program,
            num_devices,
            device_list,
            options,
            pfn_notify,
            user_data);
}

static void _init_dispatch(void) {
  dispatch.clGetDeviceInfo = &clGetDeviceInfo_wrap;
  dispatch.clCreateProgramWithBinary = &clCreateProgramWithBinary_wrap;
  dispatch.clBuildProgram = &clBuildProgram_wrap;
}
