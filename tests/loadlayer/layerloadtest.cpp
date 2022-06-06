// OpenCL includes
#include <CL/opencl.h>

// C++ Standard includes
#include <exception>
#include <iostream>
#include <vector>

bool platform_supports_spirv(cl_platform_id platform) {
  size_t ret_size;
  clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 0, nullptr, &ret_size);
  std::string extensions(ret_size, '\0');
  clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, ret_size,
                    extensions.data(), nullptr);
  return extensions.find("cl_khr_il_program") != std::string::npos;
}

bool device_supports_spirv(cl_device_id device) {
  size_t ret_size;
  clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &ret_size);
  std::string extensions(ret_size, '\0');
  clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ret_size, extensions.data(),
                  nullptr);
  return extensions.find("cl_khr_il_program") != std::string::npos;
}

std::vector<cl_platform_id> get_platforms() {
  cl_uint num_platforms;
  clGetPlatformIDs(0, nullptr, &num_platforms);
  std::vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  return platforms;
}

std::vector<cl_device_id> get_devices(cl_platform_id platform) {
  cl_uint num_devices;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
  std::vector<cl_device_id> devices(num_devices);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(),
                 nullptr);
  return devices;
}

template <int CL_INFO> std::string get_platform_info(cl_platform_id platform) {
  size_t ret_size;
  clGetPlatformInfo(platform, CL_INFO, 0, nullptr, &ret_size);
  std::string info(ret_size, '\0');
  clGetPlatformInfo(platform, CL_INFO, ret_size, info.data(), nullptr);
  info.pop_back(); // pop null-terminator
  return info;
}

template <int CL_INFO> std::string get_device_info(cl_device_id device) {
  size_t ret_size;
  clGetDeviceInfo(device, CL_INFO, 0, nullptr, &ret_size);
  std::string info(ret_size, '\0');
  clGetDeviceInfo(device, CL_INFO, ret_size, info.data(), nullptr);
  info.pop_back(); // pop null-terminator
  return info;
}

int main(int argc, char *argv[]) {
  try // Any error results in program termination
  {
    std::vector<cl_platform_id> platforms = get_platforms();

    if (platforms.empty())
      throw std::runtime_error{"No OpenCL platforms found."};

    if (argc == 1) {
      std::cout << "Found platform" << (platforms.size() > 1 ? "s" : "")
                << ":\n";
      for (const auto &platform : platforms) {
        std::cout << "\t" << get_platform_info<CL_PLATFORM_VENDOR>(platform)
                  << " (" << get_platform_info<CL_PLATFORM_VERSION>(platform)
                  << ")" << std::endl;
        std::cout << "\t\t"
                  << (platform_supports_spirv(platform)
                          ? "Supports cl_khr_il_program"
                          : "Doesn't support cl_khr_il_program")
                  << std::endl;
      }
    } else {
      cl_platform_id platform = platforms.at(argc > 1 ? std::atoi(argv[1]) : 0);
      std::cout << "Selected platform: "
                << get_platform_info<CL_PLATFORM_VENDOR>(platform) << " ("
                << get_platform_info<CL_PLATFORM_VERSION>(platform) << ")"
                << std::endl;
      std::cout << "\t\t"
                << (platform_supports_spirv(platform)
                        ? "Supports cl_khr_il_program"
                        : "Doesn't support cl_khr_il_program")
                << std::endl;

      std::vector<cl_device_id> devices = get_devices(platform);

      if (devices.empty())
        throw std::runtime_error{"No devices found on selected platform."};

      cl_device_id device = devices.at(argc > 2 ? std::atoi(argv[2]) : 0);
      std::cout << "Selected device: "
                << get_device_info<CL_DEVICE_NAME>(device) << std::endl;
      std::cout << "\t\t"
                << (device_supports_spirv(device)
                        ? "Supports cl_khr_il_program"
                        : "Doesn't support cl_khr_il_program")
                << std::endl;
    }
  } catch (std::exception &error) // If STL/CRT error occurs
  {
    std::cerr << error.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
