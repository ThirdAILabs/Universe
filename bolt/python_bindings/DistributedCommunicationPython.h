#include <pybind11/pybind11.h>
#include <bolt/src/nn/model/Model.h>


namespace py = pybind11;

namespace thirdai::bolt::train::python {
    
    class DistributedCommPython{
        public:
            explicit DistributedCommPython(py::object &py_instance);

            void communicate(const bolt::nn::model::ModelPtr& model);

            uint64_t min_num_batches(uint64_t num_batches);

        private:
            py::object py_instance;
    };
} // namespace thirdai::bolt::train::python