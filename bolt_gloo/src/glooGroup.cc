#include "glooGroup.h"
#include <stdexcept>

namespace bolt_gloo::gloo_group {

    glooGroup::glooGroup(
        uint32_t world_size, uint32_t rank, std::string &group_name, std::string &store_path, std::string &process_ip_address
    )
    :
    _world_size(world_size),
    _rank(rank),
    _group_name(std::move(group_name)) {

        auto context = gloo::rendezvous::Context(rank, world_size);
        auto file_store = createStore(store_path);

        std::string device_type = "tcp";

        auto dev = createDevice(device_type, process_ip_address);
        context.connectFullMesh(file_store, dev);
     
    }

     gloo::rendezvous::FileStore glooGroup::createStore(std::string &store_path){
       return gloo::rendezvous::FileStore(store_path);
    }


    gloo::transport::tcp::Device glooGroup::createDevice(std::string &device_type, std::string &process_ip_address){
        
        if(device_type=="tcp"){
            return   gloo::transport::tcp::CreateDevice(gloo::transport::tcp::attr(process_ip_address));
        }            
        throw std::logic_error("device_type not supported!");
       
    }

}  // namespace bolt_gloo::gloo_group