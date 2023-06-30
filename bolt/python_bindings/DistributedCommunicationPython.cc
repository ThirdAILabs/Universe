#include "DistributedCommunicationPython.h"



namespace thirdai::bolt::train::python {
    
    DistributedCommPython::DistributedCommPython(py::object &py_instance) :  py_instance(std::move(py_instance)){}

            void DistributedCommPython::communicate(const bolt::nn::model::ModelPtr& model){
                // Acquire the GIL
                py::gil_scoped_acquire acquire;

                py::object result = py_instance.attr("communicate")(model);

                // Release the GIL
                py::gil_scoped_release release;
                // Continue with other C++ computations without the GIL
            }

            uint64_t DistributedCommPython::min_num_batches(uint64_t num_batches) {
                // Acquire the GIL
                py::gil_scoped_acquire acquire;

                py::int_ result = py_instance.attr("min_num_batches")(num_batches);
                
                // Release the GIL
                py::gil_scoped_release release;
                // Continue with other C++ computations without the GIL

                uint64_t min_num_batches = static_cast<uint64_t>(result);
                return min_num_batches;
            }

    } // namespace thirdai::bolt::train::python