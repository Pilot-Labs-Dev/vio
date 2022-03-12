//
// Created by root on 4/27/21.
//
#include <vio/cl_class.h>
opencl::opencl(vk::AbstractCamera* cam):cam(cam) {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0){
        std::cerr << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    for(auto i:all_platforms){
        std::cout << "Find platform number:"<< i.getInfo<CL_PLATFORM_NAME>() << "\n";
        i.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
    }

    if (all_devices.size() == 0){
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    device = new cl::Device(all_devices[0]);
    std::cout << "CL_DEVICE_NAME: " << device->getInfo<CL_DEVICE_NAME>() <<'\n'
              << "CL_DRIVER_VERSION: " <<device->getInfo<CL_DRIVER_VERSION>()<<'\n'
              << "CL_DEVICE_OPENCL_C_VERSION: " <<device->getInfo<CL_DEVICE_OPENCL_C_VERSION>()<<'\n'
              << "CL_DEVICE_COMPILER_AVAILABLE: " <<device->getInfo<CL_DEVICE_COMPILER_AVAILABLE>()<<'\n'
              << "CL_DEVICE_LOCAL_MEM_SIZE: " <<device->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()<<'\n'
              << "CL_DEVICE_GLOBAL_MEM_SIZE: " <<device->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()<<'\n'
              << "CL_DEVICE_EXECUTION_CAPABILITIES: " <<device->getInfo<CL_DEVICE_EXECUTION_CAPABILITIES>()<<'\n'
              << "CL_DEVICE_EXTENSIONS: " <<device->getInfo<CL_DEVICE_EXTENSIONS>()<<'\n';
    context=new cl::Context({ *device });
    cl::Program::Sources sources;
    read_cl fast(std::string(KERNEL_DIR)+"/fast-gray.cl");
    sources.push_back({ fast.src_str, fast.size });
    read_cl compute_residual(std::string(KERNEL_DIR)+"/compute-residual.cl");
    sources.push_back({ compute_residual.src_str, compute_residual.size });
    program=new cl::Program(*context, sources);
    double* camera=cam->params();
    std::string options="-DFAST_THRESH=40 -DPATCH_SIZE=8 -DPATCH_HALFSIZE=4 -DF_X="+ std::to_string(camera[0]) +
                        " -DF_Y="+std::to_string(camera[1])+
                        " -DC_X="+std::to_string(camera[2])+
                        " -DC_Y="+std::to_string(camera[3])+
                        " -DS="+std::to_string(camera[4])+
                        " -DFREAK_LOG2=0.693147180559945 -DFREAK_NB_ORIENTATION=256 -DFREAK_NB_POINTS=43"+
                        " -DFREAK_SMALLEST_KP_SIZE=7 -DNB_PAIRS=512 -DNB_SCALES=64";
    if(program->build({ *device },options.c_str()) !=0)
        std::cout << " Error building: " << program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*device)<<'\n'
                  << " Binary type: " << program->getBuildInfo<CL_PROGRAM_BINARY_TYPE>(*device)<<'\n'
                  <<program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device) << '\n';
    queue=new cl::CommandQueue(*context,*device,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,NULL);
}
opencl::~opencl() {
    clear_buf();
    queue->finish();
    delete program;
    delete context;
    delete device;
}

