#include <cstdint>
#include <string>
#include <gloo/gloo/rendezvous/context.h>
#include <gloo/gloo/transport/tcp/device.h>
#include <gloo/gloo/rendezvous/file_store.h>


namespace bolt_gloo::gloo_group {
class glooGroup{

    private:
        glooGroup(uint32_t world_size, uint32_t rank, std::string &group_name, std::string &store_path, std::string &process_ip_address);

    public:

        gloo::rendezvous::FileStore createStore(std::string &store_path);
        gloo::transport::tcp::Device createDevice(std::string &device_type, std::string &process_ip_address);

    private:
        uint32_t _world_size;
        uint32_t _rank;
        std::string _group_name;
};

}  // namespace bolt_gloo::gloo_group
