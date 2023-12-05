#include "SmxPython.h"
#include <pybind11/attr.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Dtype.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Shape.h>
#include <smx/src/tensor/Tensor.h>
#include <stdexcept>

namespace thirdai::smx::python {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

class PyBufferHandle final : public MemoryHandle {
 public:
  explicit PyBufferHandle(py::buffer_info buffer)
      : _buffer(std::move(buffer)) {}

  void* ptr() const final { return _buffer.ptr; }

  size_t nbytes() const final { return _buffer.size * _buffer.itemsize; }

 private:
  py::buffer_info _buffer;
};

template <typename T>
DenseTensorPtr denseTensorFromNumpy(const NumpyArray<T>& array) {
  std::vector<size_t> shape(array.shape(), array.shape() + array.ndim());

  return DenseTensor::make(Shape(shape), getDtype<T>(),
                           std::make_shared<PyBufferHandle>(array.request()));
}

// TODO(Nicholas): is there a benefit to accepting py::buffer instead, does this
// give compatability with pytorch and tensorflow?
py::object denseTensorToNumpy(const DenseTensorPtr& tensor) {
  switch (tensor->dtype()) {
    case Dtype::f32:
      return NumpyArray<float>(/*shape=*/tensor->shape().vector(),
                               /*ptr=*/tensor->data<float>(),
                               /*base=*/py::cast(tensor));
    case Dtype::u32:
      return NumpyArray<uint32_t>(/*shape=*/tensor->shape().vector(),
                                  /*ptr=*/tensor->data<uint32_t>(),
                                  /*base=*/py::cast(tensor));
    default:
      throw std::invalid_argument("Cannot convert tensor with dtype " +
                                  toString(tensor->dtype()) + " to numpy.");
  }
}

void defineTensor(py::module_& smx) {
  py::class_<Shape>(smx, "Shape")
      .def(py::init<std::vector<size_t>>())
      .def(py::init([](const py::args& dims) {
        std::vector<size_t> shape;
        for (const auto& dim : dims) {
          shape.push_back(dim.cast<size_t>());
        }
        return Shape(shape);
      }))
      .def("__getitem__", &Shape::operator[])
      .def("__len__", &Shape::ndim)
      .def(
          "__iter__",
          [](const Shape& shape) {
            return py::make_iterator(shape.begin(), shape.end());
          },
          py::keep_alive<0, 1>())
      .def("__str__", &Shape::toString);

  py::enum_<Dtype>(smx, "Dtype")
      .value("f32", Dtype::f32)
      .value("u32", Dtype::u32)
      .export_values();

  py::class_<Tensor, TensorPtr>(smx, "Tensor")
      .def_property_readonly("shape", &Tensor::shape)
      .def_property_readonly("dtype", &Tensor::dtype)
      .def_property_readonly("ndim", &Tensor::ndim);

  py::class_<DenseTensor, DenseTensorPtr, Tensor>(smx, "DenseTensor")
      .def_property_readonly("strides", &DenseTensor::strides)
      .def("numpy", &denseTensorToNumpy);

  smx.def("transpose", &transpose, py::arg("tensor"), py::arg("perm"));
  smx.def("reshape", &reshape, py::arg("tensor"), py::arg("new_shape"));

  smx.def("from_numpy", &denseTensorFromNumpy<float>, py::arg("array"));
  smx.def("from_numpy", &denseTensorFromNumpy<uint32_t>, py::arg("array"));
}

void createSmxSubmodule(py::module_& mod) {
  auto smx = mod.def_submodule("smx");

  defineTensor(smx);
}

}  // namespace thirdai::smx::python