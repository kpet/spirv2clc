#pragma once

#if defined(_WIN32)
#define NOMINMAX
#endif

#include <CL/cl_layer.h>

CL_API_ENTRY cl_int CL_API_CALL
clGetLayerInfo(
  cl_layer_info param_name,
  size_t        param_value_size,
  void*         param_value,
  size_t*       param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL
clInitLayer(
  cl_uint                  num_entries,
  const _cl_icd_dispatch*  target_dispatch,
  cl_uint*                 num_entries_out,
  const _cl_icd_dispatch** layer_dispatch_ret);

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo_wrap(
  cl_device_id device,
  cl_device_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret);

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithIL_wrap(
  cl_context context,
  const void* il,
  size_t length,
  cl_int* errcode_ret);

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram_wrap(
  cl_program program,
  cl_uint num_devices,
  const cl_device_id* device_list,
  const char* options,
  void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
  void* user_data);

#include <map>
#include <memory>
#include <string>

namespace spirv2clc
{
  class layer
  {
  public:
    layer(const _cl_icd_dispatch* target_dispatch) noexcept;

    layer() = delete;
    layer(const layer&) = delete;
    layer(layer&&) = delete;
    ~layer() = default;
    layer& operator=(const layer&) = delete;
    layer& operator=(layer&&) = delete;

    const _cl_icd_dispatch& get_dispatch() const;
    const _cl_icd_dispatch& get_target_dispatch() const;

    cl_int clGetDeviceInfo(
      cl_device_id device,
      cl_device_info param_name,
      size_t param_value_size,
      void* param_value,
      size_t* param_value_size_ret);

    cl_program clCreateProgramWithIL(
      cl_context context,
      const void* il,
      size_t length,
      cl_int* errcode_ret);

    cl_int clBuildProgram(
      cl_program program,
      cl_uint num_devices,
      const cl_device_id* device_list,
      const char* options,
      void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
      void* user_data);

    std::map<cl_program, std::string> program_ils;

  private:
    void init_dispatch(void);

    bool device_supports_spirv_via_extension(
      cl_device_id device,
      cl_int* err);

    bool spirv_queries_are_core_for_device(
      cl_device_id device,
      cl_int* err);

    bool spirv_queries_are_valid_for_device(
      cl_device_id device,
      cl_int* err);

    bool device_supports_spirv_out_of_the_box(
      cl_device_id device,
      cl_int* err);

    cl_int clGetDeviceInfo_CL_DEVICE_EXTENSIONS(
      cl_device_id device,
      cl_device_info param_name,
      size_t param_value_size,
      void* param_value,
      size_t* param_value_size_ret);

    cl_int clGetDeviceInfo_CL_DEVICE_IL_VERSION(
      cl_device_id device,
      cl_device_info param_name,
      size_t param_value_size,
      void* param_value,
      size_t* param_value_size_ret);

    cl_int clGetDeviceInfo_CL_DEVICE_ILS_WITH_VERSION(
      cl_device_id device,
      cl_device_info param_name,
      size_t param_value_size,
      void* param_value,
      size_t* param_value_size_ret);

    const _cl_icd_dispatch* tdispatch;
    _cl_icd_dispatch dispatch;
  };

  inline std::unique_ptr<layer> instance;
}
