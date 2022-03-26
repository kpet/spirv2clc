#include "layer.hpp"

CL_API_ENTRY cl_int CL_API_CALL
clGetLayerInfo(
  cl_layer_info  param_name,
  size_t         param_value_size,
  void          *param_value,
  size_t        *param_value_size_ret)
{
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
clInitLayer(
  cl_uint                         num_entries,
  const struct _cl_icd_dispatch  *target_dispatch,
  cl_uint                        *num_entries_out,
  const struct _cl_icd_dispatch **layer_dispatch_ret)
{
  if (!target_dispatch || 
      !layer_dispatch_ret ||
      !num_entries_out || 
      num_entries < sizeof(_cl_icd_dispatch) / sizeof(_cl_icd_dispatch::clGetPlatformIDs)
  )
    return CL_INVALID_VALUE;

  spirv2clc::instance = std::make_unique<spirv2clc::layer>(target_dispatch);

  *layer_dispatch_ret = &spirv2clc::instance->get_dispatch();
  *num_entries_out = sizeof(_cl_icd_dispatch) / sizeof(_cl_icd_dispatch::clGetPlatformIDs);

  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo_wrap(
  cl_device_id device,
  cl_device_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  return spirv2clc::instance->clGetDeviceInfo(
    device,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret
  );
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithIL_wrap(
  cl_context context,
  const void* il,
  size_t length,
  cl_int* errcode_ret)
{
  return spirv2clc::instance->clCreateProgramWithIL(
    context,
    il,
    length,
    errcode_ret
  );
}

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram_wrap(
  cl_program program,
  cl_uint num_devices,
  const cl_device_id* device_list,
  const char* options,
  void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
  void* user_data)
{
  return spirv2clc::instance->clBuildProgram(
    program,
    num_devices,
    device_list,
    options,
    pfn_notify,
    user_data);
}

CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo_wrap(
  cl_program program,
  cl_program_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  return spirv2clc::instance->clGetProgramInfo(
    program,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret
  );
}

#include "spirv2clc.h"

#include <vector>
#include <map>
#include <string_view>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <regex>

namespace spirv2clc
{

layer::layer(const _cl_icd_dispatch* target_dispatch) noexcept
  : tdispatch(target_dispatch)
  , dispatch()
{
  init_dispatch();
}

void layer::init_dispatch(void)
{
  dispatch.clGetDeviceInfo = &clGetDeviceInfo_wrap;
  dispatch.clCreateProgramWithIL = &clCreateProgramWithIL_wrap;
  dispatch.clBuildProgram = &clBuildProgram_wrap;
}

const _cl_icd_dispatch& layer::get_dispatch() const
{
  return dispatch;
}

const _cl_icd_dispatch& layer::get_target_dispatch() const
{
  return *tdispatch;
}

bool layer::device_supports_spirv_via_extension(
  cl_device_id device,
  cl_int* err
)
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
    if (err_ == CL_SUCCESS)
    {
      std::string_view dispatch_extensions_string{
        dispatch_extensions.data(),
        dispatch_extensions.size()
      };
      static const char* spirv_ext_name = "cl_khr_il_program";
      if (dispatch_extensions_string.find(spirv_ext_name) != std::string::npos)
        return true;
      else
        return false;
    }
    else
    {
      if (err != nullptr) *err = err_;
      return false;
    }
  }
  else
  {
    if (err != nullptr) *err = err_;
    return false;
  }
}

bool layer::spirv_queries_are_core_for_device(
  cl_device_id device,
  cl_int* err
)
{
  cl_platform_id device_platform;
  cl_int err_ = tdispatch->clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &device_platform, nullptr);
  if (err_ == CL_SUCCESS)
  {
    size_t platform_version_size_ret = 0;
    err_ = err_ = tdispatch->clGetPlatformInfo(device_platform, CL_PLATFORM_VERSION, 0, nullptr, &platform_version_size_ret);
    if (err_ == CL_SUCCESS)
    {
      std::string platform_version(
      platform_version_size_ret,
      '\0'
      );
      err_ = tdispatch->clGetPlatformInfo(device_platform, CL_PLATFORM_VERSION, platform_version.size(), platform_version.data(), nullptr);
      if (err_ == CL_SUCCESS)
      {
        if(platform_version.find("OpenCL 2.1") != std::string::npos ||
           platform_version.find("OpenCL 2.2") != std::string::npos)
        {
          if (err != nullptr) *err = CL_SUCCESS;
          return true;
        }
        else
        {
          if (err != nullptr) *err = CL_SUCCESS;
          return false;
        }
      }
      else
      {
        if (err != nullptr) *err = CL_SUCCESS;
        return false;
      }
    }
    else
    {
      if (err != nullptr) *err = CL_SUCCESS;
      return true;
    }
  }
  else
  {
    if (err != nullptr) *err = CL_SUCCESS;
    return false;
  }
}

bool layer::spirv_queries_are_valid_for_device(
  cl_device_id device,
  cl_int* err
)
{
  if (spirv_queries_are_core_for_device(device, err))
    return true;
  else
    return device_supports_spirv_via_extension(device, err);
}

bool layer::device_supports_spirv_out_of_the_box(
  cl_device_id device,
  cl_int* err
)
{
  if (spirv_queries_are_valid_for_device(device, err))
  {
    size_t il_version_size_ret;
    cl_int err_ = tdispatch->clGetDeviceInfo(
      device,
      CL_DEVICE_IL_VERSION,
      0,
      nullptr,
      &il_version_size_ret
    );
    if (err_ == CL_SUCCESS)
    {
      std::string il_version(il_version_size_ret, '\0');
      err_ = tdispatch->clGetDeviceInfo(device, CL_DEVICE_IL_VERSION, il_version.size(), il_version.data(), nullptr);
      if (err_ == CL_SUCCESS)
      {
        if(il_version.find("SPIR-V") != std::string::npos)
        {
          if (err != nullptr) *err = CL_SUCCESS;
          return true;
        }
        else
        {
          if (err != nullptr) *err = CL_SUCCESS;
          return false;
        }
      }
      else
      {
        if (err != nullptr) *err = err_;
        return false;
      }
    }
    else
    {
      if (err != nullptr) *err = err_;
      return false;
    }
  }
  else
  {
    return false;
  }
}

cl_int layer::clGetDeviceInfo(
  cl_device_id device,
  cl_device_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  switch (param_name)
  {
    case CL_DEVICE_EXTENSIONS:
      return clGetDeviceInfo_CL_DEVICE_EXTENSIONS(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret
      );
      break;
    case CL_DEVICE_IL_VERSION:
      return clGetDeviceInfo_CL_DEVICE_IL_VERSION(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret
      );
      break;
    case CL_DEVICE_ILS_WITH_VERSION:
      return clGetDeviceInfo_CL_DEVICE_ILS_WITH_VERSION(
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

cl_program layer::clCreateProgramWithIL(
  cl_context context,
  const void* il,
  size_t length,
  cl_int* errcode_ret)
{
  spirv2clc::translator translator;
  std::string srcgen;
  std::vector<uint32_t> binary(
    static_cast<const uint32_t*>(il),
    static_cast<const uint32_t*>(il) + length / sizeof(uint32_t));
  int err = translator.translate(binary, &srcgen);

  if (err == 0)
  {
    const char* str = srcgen.data();
    size_t len = srcgen.length();
    cl_program prog = tdispatch->clCreateProgramWithSource(
      context,
      1,
      &str,
      &len,
      errcode_ret);
    if (*errcode_ret == CL_SUCCESS)
    {
      program_ils[prog] =
        std::make_pair(
          std::string(static_cast<const char*>(il), length),
          std::string(str, len)
        );
    }
    return prog;
  }
  else
  {
    return tdispatch->clCreateProgramWithIL(
    context,
    il,
    length,
    errcode_ret);
  }
}

cl_int layer::clBuildProgram(
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

cl_int layer::clGetProgramInfo(
  cl_program program,
  cl_device_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  switch (param_name)
  {
    case CL_PROGRAM_IL:
      return clGetProgramInfo_CL_PROGRAM_IL(
        program,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
      break;
    default:
      return tdispatch->clGetProgramInfo(
        program,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
      break;
  }
}

cl_int layer::clGetDeviceInfo_CL_DEVICE_EXTENSIONS(
  cl_device_id device,
  cl_device_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  cl_int err_;
  bool layer_has_nothing_to_do = device_supports_spirv_out_of_the_box(device, &err_);
  if (layer_has_nothing_to_do || err_ != CL_SUCCESS)
  {
    return tdispatch->clGetDeviceInfo(
      device,
      CL_DEVICE_EXTENSIONS,
      param_value_size,
      param_value,
      param_value_size_ret);
  }
  else
  {
    bool layer_has_work_but_need_not_alter_extension_string =
      spirv_queries_are_core_for_device(device, &err_);
    if (layer_has_work_but_need_not_alter_extension_string || err_ != CL_SUCCESS)
    {
      return tdispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_EXTENSIONS,
        param_value_size,
        param_value,
        param_value_size_ret);
    }
    else
    {
      size_t dispatch_param_value_size_ret = 0;
      cl_int err_ = tdispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_EXTENSIONS,
        0,
        nullptr,
        &dispatch_param_value_size_ret);
      if (err_ == CL_SUCCESS)
      {
        std::string dispatch_extensions(
          dispatch_param_value_size_ret - 1,
          '\0'
        );
        err_ = tdispatch->clGetDeviceInfo(
          device,
          CL_DEVICE_EXTENSIONS,
          dispatch_extensions.size() + 1,
          dispatch_extensions.data(),
          nullptr
        );
        dispatch_extensions.append(" cl_khr_il_program");

        if (param_value_size_ret != nullptr)
          *param_value_size_ret = dispatch_extensions.length() * sizeof(char) + 1;

        if (param_value_size >= dispatch_extensions.length() * sizeof(char))
        {
          if (param_value != nullptr)
          {
            std::copy(
              dispatch_extensions.begin(),
              dispatch_extensions.end(),
              static_cast<char*>(param_value)
            );
            return CL_SUCCESS;
          }
          else
            return CL_INVALID_VALUE;
        }
        else if (param_value != nullptr)
          return CL_INVALID_VALUE;
      }
      else
      {
        return err_;
      }
    }
  }
}

cl_int layer::clGetDeviceInfo_CL_DEVICE_IL_VERSION(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  cl_int err_;
  bool layer_has_nothing_to_do = device_supports_spirv_out_of_the_box(device, &err_);
  if (layer_has_nothing_to_do || err_ != CL_SUCCESS)
  {
    return tdispatch->clGetDeviceInfo(
      device,
      CL_DEVICE_IL_VERSION,
      param_value_size,
      param_value,
      param_value_size_ret);
  }
  else
  {
    size_t dispatch_param_value_size_ret = 0;
    cl_int err_ = tdispatch->clGetDeviceInfo(
      device,
      CL_DEVICE_IL_VERSION,
      0,
      nullptr,
      &dispatch_param_value_size_ret);
    if (err_ == CL_SUCCESS)
    {
      std::string dispatch_il_versions(
        dispatch_param_value_size_ret,
        '\0'
      );
      err_ = tdispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_EXTENSIONS,
        dispatch_il_versions.size(),
        dispatch_il_versions.data(),
        nullptr
      );
      dispatch_il_versions.append(" SPIR-V_1.2");

      if (param_value_size_ret != nullptr)
        *param_value_size_ret = dispatch_il_versions.length() * sizeof(char);

      if (param_value_size >= dispatch_il_versions.length() * sizeof(char))
      {
        if (param_value != nullptr)
        {
          std::copy(
            dispatch_il_versions.begin(),
            dispatch_il_versions.end(),
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
    else
    {
      return err_;
    }
  }
}

cl_int layer::clGetDeviceInfo_CL_DEVICE_ILS_WITH_VERSION(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  cl_int err_;
  bool layer_has_nothing_to_do = device_supports_spirv_out_of_the_box(device, &err_);
  if (layer_has_nothing_to_do || err_ != CL_SUCCESS)
  {
    return tdispatch->clGetDeviceInfo(
      device,
      CL_DEVICE_ILS_WITH_VERSION,
      param_value_size,
      param_value,
      param_value_size_ret);
  }
  else
  {
    size_t dispatch_param_value_size_ret = 0;
    cl_int err_ = tdispatch->clGetDeviceInfo(
      device,
      CL_DEVICE_ILS_WITH_VERSION,
      0,
      nullptr,
      &dispatch_param_value_size_ret);
    if (err_ == CL_SUCCESS)
    {
      std::vector<cl_name_version> dispatch_il_versions(
        dispatch_param_value_size_ret
      );
      err_ = tdispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_ILS_WITH_VERSION,
        dispatch_il_versions.size(),
        dispatch_il_versions.data(),
        nullptr
      );
      dispatch_il_versions.push_back(cl_name_version{
        cl_version{CL_MAKE_VERSION(1, 2, 0)},
        "SPIR-V"
      });

      if (param_value_size_ret != nullptr)
        *param_value_size_ret = dispatch_il_versions.size() * sizeof(cl_name_version);

      if (param_value_size >= dispatch_il_versions.size() * sizeof(cl_name_version))
      {
        if (param_value != nullptr)
        {
          std::copy(
            dispatch_il_versions.begin(),
            dispatch_il_versions.end(),
            static_cast<cl_name_version*>(param_value)
          );
          return CL_SUCCESS;
        }
        else
          return CL_INVALID_VALUE;
      }
      else
        return CL_INVALID_VALUE;
    }
    else
    {
      return err_;
    }
  }
}

cl_int layer::clGetProgramInfo_CL_PROGRAM_IL(
      cl_program program,
      cl_program_info param_name,
      size_t param_value_size,
      void* param_value,
      size_t* param_value_size_ret)
{
  if (param_value_size_ret != nullptr)
    *param_value_size_ret = program_ils[program].first.length() * sizeof(char);

  if (param_value_size >= program_ils[program].first.length() * sizeof(char))
  {
    if (param_value != nullptr)
    {
      std::copy(
        program_ils[program].first.begin(),
        program_ils[program].first.end(),
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

} // namespace spirv2clc
