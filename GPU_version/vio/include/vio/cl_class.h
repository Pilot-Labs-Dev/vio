//
// Created by root on 4/27/21.
//

#ifndef VIO_OPENCL_CL_CLASS_H
#define VIO_OPENCL_CL_CLASS_H
#include <vio/cl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <exception>
#include <CL/opencl.h>
#include <opencv2/opencv.hpp>
#include <vio/abstract_camera.h>
#include <vio/for_it.hpp>

class kernel{
public:
    kernel(cl::Program* program,std::string name){
        cl_int error;
        _kernel = new cl::Kernel(*program,name.c_str(),&error);
        assert(error== CL_SUCCESS);
    };
    ~kernel(){
        for(auto&& i:_images)i.first->setDestructorCallback((void (*)(_cl_mem *, void *))notify, NULL);
        for(auto&& i:_buffers)i.first->setDestructorCallback((void (*)(_cl_mem *, void *))notify, NULL);
        _buffers.clear();
        _images.clear();
    };
    template<typename T>
    void load(size_t id/*buffer ID*/,T* buf,cl::CommandQueue* queue,cl::Context* context,size_t buf_size){
        assert(buf);
        cl_int error;
        _buffers.push_back(std::pair<std::shared_ptr<cl::Buffer>,size_t>(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(T) * buf_size,buf,&error),id));
        assert(error == CL_SUCCESS);
        cl_int err=_kernel->setArg(id,*_buffers.back().first);
        assert(err == CL_SUCCESS);

    };
    void load(size_t id/*buffer ID*/,cv::Mat& buf,cl::Context* context){
        assert(!buf.empty());
        cl_int error;
        _images.push_back(std::pair<std::shared_ptr<cl::Image2D>,size_t>(std::make_shared<cl::Image2D>(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                                                    cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
                                                                    buf.size().width,
                                                                    buf.size().height,
                                                                    0,
                                                                    reinterpret_cast<uchar*>(buf.data),&error),id));
        assert(error == CL_SUCCESS);
        cl_int err=_kernel->setArg(id,*_images.back().first);
        assert(err == CL_SUCCESS);
    };
    template<typename T>
    void reload(size_t id,T* buf,cl::CommandQueue* queue,size_t size){
        assert(buf);
        cl_int error;
        for(auto&& i:_buffers)if(i.second==id){
                cl::Event event;
                T* Map_buf=(T*)queue->enqueueMapBuffer(*i.first,CL_NON_BLOCKING,CL_MAP_WRITE,0,sizeof(T) * size,NULL,&event,&error);
                event.wait();
                memcpy(Map_buf,buf,sizeof(T) * size);
                assert(queue->enqueueUnmapMemObject(*i.first,Map_buf,NULL,&event)==CL_SUCCESS);
                event.wait();

            }
        assert(error == CL_SUCCESS);
    };
    void release(size_t id/*buffer ID*/){
        for(auto it=_buffers.begin();it!=_buffers.end();++it){
            if((*it).second==id){
                (*it).first->setDestructorCallback((void (*)(_cl_mem *, void *))notify, NULL);
                (*it).first.reset();
                _buffers.erase(it);
                return;
            }
        }
        for(auto it=_images.begin();it!=_images.end();++it){
            if((*it).second==id){
                (*it).first->setDestructorCallback((void (*)(_cl_mem *, void *))notify, NULL);
                (*it).first.reset();
                _images.erase(it);
                return;
            }
        }
    };
    void release(){
        _buffers.clear();
        _images.clear();
    }
    cl::Kernel* _kernel = NULL;
    std::shared_ptr<cl::Buffer> read(size_t id){
        for(auto&& i:_buffers)if(i.second==id)return i.first;
    }
private:
    static void notify(cl_mem *, void * user_data) {
    }
    std::list<std::pair<std::shared_ptr<cl::Buffer>,size_t>> _buffers;
    std::list<std::pair<std::shared_ptr<cl::Image2D>,size_t>> _images;

};
class read_cl{
public:
    read_cl(std::string path){
        src_str=(char *)calloc(0x100000, sizeof(char));
        FILE *fp;
        fp = fopen(path.c_str(), "r");
        size = fread(src_str, 1, 0x100000, fp);
        fclose(fp);
    }
    void print(){
        std::cerr<<src_str<<"\n";
    }
    char *src_str = nullptr;
    size_t size;
};
class opencl{
public:
    opencl(vk::AbstractCamera* cam);
    ~opencl();
    int32_t make_kernel(std::string name){_kernels.push_back(kernel(program,name));};
    template<typename T>
    void load(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/, int size,T* buf) {
        _kernels.at(id1).load(id2,buf,queue,context,size);
    }
    void load(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,cv::Mat& buf) {
        _kernels.at(id1).load(id2,buf,context);
    }
    template<typename T>
    void load(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,T&& value){
        _kernels.at(id1)._kernel->setArg(id2,value);
    };
    template<typename T>
    void reload(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/, size_t size,T* buf){
        _kernels.at(id1).reload(id2,buf,queue,size);
    };
    cl_int run(size_t id1/*kernal ID*/,std::size_t  x=1,std::size_t y=1,std::size_t z=1) {
        cl_int err=0;
        cl::Event event;
        assert(queue->flush()==CL_SUCCESS);
        if(z>1 && y>1){
            err=queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x,y,z)/*Global*/, cl::NullRange/*local*/,NULL,&event);
        }else if(z<2 && y>1){
            err=queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x,y)/*Global*/, cl::NullRange/*local*/,NULL,&event);
        }else{
            err=queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x)/*Global*/, cl::NullRange/*local*/,NULL,&event);
        };
        assert(err==CL_SUCCESS);
        event.wait();
        assert(queue->finish()==CL_SUCCESS);
        return 1;

    }
    template<typename T>
    void read(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,size_t size/*size*/, T* out){
        assert(out);
        cl_int error;
        cl::Event event;
        T* Map_buf=(T*)queue->enqueueMapBuffer(*_kernels.at(id1).read(id2),CL_NON_BLOCKING,CL_MAP_READ,0,sizeof(T) * size,NULL,&event,&error);
        assert(error == CL_SUCCESS);
        event.wait();
        memcpy(out,Map_buf,sizeof(T) * size);
        assert(queue->enqueueUnmapMemObject(*_kernels.at(id1).read(id2),Map_buf,NULL,&event)==CL_SUCCESS);
        event.wait();
    }
    void clear_buf(){
        for(auto k:_kernels)k.release();
    };
    void release(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/){
        _kernels.at(id1).release(id2);
    }
private:
    std::vector<kernel> _kernels;
    cl::Context* context = nullptr;
    cl::Device* device = nullptr;
    cl::Program* program = nullptr;
    cl::CommandQueue* queue= nullptr;
    vk::AbstractCamera* cam= nullptr;
};

#endif //VIO_OPENCL_CL_CLASS_H
