// OpenCL includes
#include <CL/opencl.hpp>

// C++ Standard includes
#include <vector>
#include <iostream>
#include <exception>

bool platform_supports_spirv(cl::Platform platform)
{
	return platform.getInfo<CL_PLATFORM_EXTENSIONS>().find("cl_khr_il_program") != std::string::npos;
}

bool device_supports_spirv(cl::Device device)
{
	return device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_il_program") != std::string::npos;
}

int main(int argc, char* argv[])
{
	try // Any error results in program termination
	{
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		if (platforms.empty()) throw std::runtime_error{ "No OpenCL platforms found." };

		if (argc == 1)
		{
			std::cout << "Found platform" << (platforms.size() > 1 ? "s" : "") << ":\n";
			for (const auto& platform : platforms)
			{
				std::cout <<
					"\t" << platform.getInfo<CL_PLATFORM_VENDOR>() <<
					" (" <<
					platform.getInfo<CL_PLATFORM_VERSION>() <<
					")" <<
					std::endl;
				std::cout <<
					"\t\t" <<
					(
						platform_supports_spirv(platform) ?
							"Supports cl_khr_il_program" :
							"Doesn't support cl_khr_il_program"
					) <<
					std::endl;
			}
		}
		else
		{
			cl::Platform platform = platforms.at(argc > 1 ? std::atoi(argv[1]) : 0);
			std::cout <<
				"Selected platform: " <<
				platform.getInfo<CL_PLATFORM_VENDOR>() <<
				" (" <<
				platform.getInfo<CL_PLATFORM_VERSION>() <<
				")" <<
				std::endl;
			std::cout <<
				"\t\t" <<
				(
					platform_supports_spirv(platform) ?
						"Supports cl_khr_il_program" :
						"Doesn't support cl_khr_il_program"
				) <<
				std::endl;

			std::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

			if (devices.empty()) throw std::runtime_error{ "No devices found on selected platform." };
/*
			std::cout << "Found device" << (devices.size() > 1 ? "s" : "") << ":" << std::endl;
			for (const auto& device : devices)
			{
				std::cout <<
					"\t" << device.getInfo<CL_DEVICE_NAME>() <<
					std::endl;
				std::cout <<
					"\t\t" <<
					(
						device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_il_program") != std::string::npos ?
							"Supports cl_khr_il_program" :
							"Doesn't support cl_khr_il_program"
					) <<
					std::endl;
			}
*/
			cl::Device device = devices.at(argc > 2 ? std::atoi(argv[2]) : 0);
			std::cout <<
				"Selected device: " <<
				device.getInfo<CL_DEVICE_NAME>() <<
				std::endl;
			std::cout <<
				"\t\t" <<
				(
					device_supports_spirv(device) ?
						"Supports cl_khr_il_program" :
						"Doesn't support cl_khr_il_program"
				) <<
				std::endl;
			}
	}
	catch (cl::Error& error) // If any OpenCL error occurs
	{
		std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
		std::exit(error.err());
	}
	catch (std::exception& error) // If STL/CRT error occurs
	{
		std::cerr << error.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	return EXIT_SUCCESS;
}
