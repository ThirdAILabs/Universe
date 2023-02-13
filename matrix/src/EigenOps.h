#include "wrappers/src/EigenDenseWrapper.h"
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Core/util/Macros.h>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorTraits.h>
#include <stdexcept>
#include <string>

using RowMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using RowTensor3DFloat = Eigen::Tensor<float, 3, Eigen::RowMajor>;
using RowTensor4DFloat = Eigen::Tensor<float, 4, Eigen::RowMajor>;

inline RowMatrixXf eigenMult(const Eigen::Ref<const RowMatrixXf>& m1,
                             const Eigen::Ref<const RowMatrixXf>& m2) {
  return m1 * m2;
}



/** 
 * Applies a 2D convolution over a multichannel input image.
 *
 * The input parameter is expected to be a tensor with a rank of 3 or more
 * (channels, height, width, and optionally others)
 * The kernel parameter is expected to be a 4D tensor (filters, channels,
 * kernel_height, kernel_width)
 * The input and the kernel must both be in col-major layout. The result will
 * also be in col-major layout.
 *
 * If col_in_stride, row_in_stride > 1, then applies convolution with holes
 * (aka atrous convolution), sampling every col_in_stride, row_in_stride input
 * pixels.
 *
 * If padding_top, padding_bottom, padding_left, or padding_right is specified,
 * then those paddings will be used to pad the input, and padding_type must be
 * PADDING_VALID.
 *
 * The result can be assigned to a tensor of rank equal to the rank of the
 * input. The dimensions of the result will be filters, height, width (and
 * others if applicable).
 *
 * It is possible to swap the order of the width and height dimensions provided
 * that the same order is used in the input, the kernel, and the output.
 *
 * It is also possible to add an output kernel to the contraction, output
 * kernel is called by Eigen when it "finalizes" the block of an output tensor.
 *
 */
template <typename Input, typename Kernel,
          typename OutputKernel = const Eigen::NoOpOutputKernel>
EIGEN_ALWAYS_INLINE std::conditional_t<
    Eigen::internal::traits<Input>::Layout == Eigen::ColMajor,
    Eigen::TensorReshapingOp<
        const Eigen::DSizes<typename Eigen::internal::traits<Input>::Index,
                     Eigen::internal::traits<Input>::NumDimensions>,
        const Eigen::TensorContractionOp<
            const Eigen::array<Eigen::IndexPair<typename Eigen::internal::traits<Input>::Index>, 1>,
            const Eigen::TensorReshapingOp<
                const Eigen::DSizes<typename Eigen::internal::traits<Input>::Index, 2>,
                const Kernel>,
            const Eigen::TensorReshapingOp<
                const Eigen::DSizes<typename Eigen::internal::traits<Input>::Index, 2>,
                const Eigen::TensorImagePatchOp<Eigen::Dynamic, Eigen::Dynamic, const Input> >,
            const OutputKernel> >,
    Eigen::TensorReshapingOp<
        const Eigen::DSizes<typename Eigen::internal::traits<Input>::Index,
                     Eigen::internal::traits<Input>::NumDimensions>,
        const Eigen::TensorContractionOp<
            const Eigen::array<Eigen::IndexPair<typename Eigen::internal::traits<Input>::Index>, 1>,
            const Eigen::TensorReshapingOp<
                const Eigen::DSizes<typename Eigen::internal::traits<Input>::Index, 2>,
                const Eigen::TensorImagePatchOp<Eigen::Dynamic, Eigen::Dynamic, const Input> >,
            const Eigen::TensorReshapingOp<
                const Eigen::DSizes<typename Eigen::internal::traits<Input>::Index, 2>,
                const Kernel>,
            const OutputKernel> > >
SpatialConvolution(const Input& input, const Kernel& kernel,
                   const Eigen::Index row_stride = 1, const Eigen::Index col_stride = 1,
                   const Eigen::PaddingType padding_type = Eigen::PADDING_SAME,
                   const Eigen::Index row_in_stride = 1, const Eigen::Index col_in_stride = 1,
                   const OutputKernel& output_kernel = OutputKernel(),
                   Eigen::Index padding_top = 0, Eigen::Index padding_bottom = 0,
                   Eigen::Index padding_left = 0, Eigen::Index padding_right = 0) {
       
  typedef typename Eigen::internal::traits<Input>::Index TensorIndex;
  typedef typename Eigen::internal::traits<Input>::Scalar InputScalar;
  Eigen::TensorRef<Eigen::Tensor<InputScalar, Eigen::internal::traits<Input>::NumDimensions,
                   Eigen::internal::traits<Input>::Layout, TensorIndex> >
      in(input);
  Eigen::TensorRef<Eigen::Tensor<typename Eigen::internal::traits<Kernel>::Scalar,
                   Eigen::internal::traits<Kernel>::NumDimensions,
                   Eigen::internal::traits<Kernel>::Layout, TensorIndex> >
      kern(kernel);

  EIGEN_STATIC_ASSERT(
      Eigen::internal::traits<Input>::Layout == Eigen::internal::traits<Kernel>::Layout,
      YOU_MADE_A_PROGRAMMING_MISTAKE)
  const bool isColMajor = (Eigen::internal::traits<Input>::Layout == Eigen::ColMajor);

  const int NumDims = Eigen::internal::traits<Input>::NumDimensions;

  // Number of filters to apply. This is the same as the output depth of the
  // result
  const TensorIndex kernelFilters =
      isColMajor ? kern.dimensions()[0] : kern.dimensions()[3];
  // Number of channels. This is the same as the input depth.
  const TensorIndex kernelChannels =
      isColMajor ? kern.dimensions()[1] : kern.dimensions()[2];
  const TensorIndex kernelRows =
      isColMajor ? kern.dimensions()[2] : kern.dimensions()[1];
  const TensorIndex kernelCols =
      isColMajor ? kern.dimensions()[3] : kern.dimensions()[0];

  const Eigen::Index kernelRowsEff =
      kernelRows + (kernelRows - 1) * (row_in_stride - 1);
  const Eigen::Index kernelColsEff =
      kernelCols + (kernelCols - 1) * (col_in_stride - 1);

  Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

  const TensorIndex InputRows =
      isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex InputCols =
      isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);
  const bool padding_explicit =
      (padding_top || padding_bottom || padding_left || padding_right);

  TensorIndex out_height;
  TensorIndex out_width;
  switch (padding_type) {
    case Eigen::PADDING_VALID: {
      const TensorIndex InputRowsEff = InputRows + padding_top + padding_bottom;
      const TensorIndex InputColsEff = InputCols + padding_left + padding_right;
      out_height = Eigen::divup(InputRowsEff - kernelRowsEff + 1, row_stride);
      out_width = Eigen::divup(InputColsEff - kernelColsEff + 1, col_stride);
      break;
    }
    case Eigen::PADDING_SAME: {
      eigen_assert(!padding_explicit);
      out_height = Eigen::divup(InputRows, row_stride);
      out_width = Eigen::divup(InputCols, col_stride);
      break;
    }
    default: {
      // Initialize unused variables to avoid a compiler warning
      out_height = 0;
      out_width = 0;
      eigen_assert(false && "unexpected padding"); // NOLINT
    }
  }

  // Molds the output of the patch extraction code into a 2d tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  // kernels
  // - the second dimension (dims[1]): everything else
  Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[1] = out_height * out_width;
    for (int i = 3; i < NumDims; ++i) {
      pre_contract_dims[1] *= in.dimension(i);
    }
  } else {
    pre_contract_dims[1] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[0] = out_height * out_width;
    for (int i = 0; i < NumDims - 3; ++i) {
      pre_contract_dims[0] *= in.dimension(i);
    }
  }

  // Molds the output of the contraction into the shape expected by the used
  // (assuming this is Eigen::ColMajor):
  // - 1st dim: kernel filters
  // - 2nd dim: output height
  // - 3rd dim: output width
  // - 4th dim and beyond: everything else including batch size
  Eigen::DSizes<TensorIndex, NumDims> post_contract_dims;
  if (isColMajor) {
    post_contract_dims[0] = kernelFilters;
    post_contract_dims[1] = out_height;
    post_contract_dims[2] = out_width;
    for (int i = 3; i < NumDims; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  } else {
    post_contract_dims[NumDims - 1] = kernelFilters;
    post_contract_dims[NumDims - 2] = out_height;
    post_contract_dims[NumDims - 3] = out_width;
    for (int i = 0; i < NumDims - 3; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  }

  Eigen::DSizes<TensorIndex, 2> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters;
    kernel_dims[1] = kernelChannels * kernelRows * kernelCols;
  } else {
    kernel_dims[0] = kernelChannels * kernelRows * kernelCols;
    kernel_dims[1] = kernelFilters;
  }
  if (padding_explicit) {
    return Eigen::choose(
        Eigen::Cond<Eigen::internal::traits<Input>::Layout == Eigen::ColMajor>(),
        kernel.reshape(kernel_dims)
            .contract(input
                          .extract_image_patches(
                              kernelRows, kernelCols, row_stride, col_stride,
                              row_in_stride, col_in_stride,
                              /*row_inflate_stride=*/1,
                              /*col_inflate_stride=*/1, padding_top,
                              padding_bottom, padding_left, padding_right,
                              /*padding_value=*/static_cast<InputScalar>(0))
                          .reshape(pre_contract_dims),
                      contract_dims, output_kernel)
            .reshape(post_contract_dims),
        input
            .extract_image_patches(
                kernelRows, kernelCols, row_stride, col_stride, row_in_stride,
                col_in_stride,
                /*row_inflate_stride=*/1,
                /*col_inflate_stride=*/1, padding_top, padding_bottom,
                padding_left, padding_right,
                /*padding_value=*/static_cast<InputScalar>(0))
            .reshape(pre_contract_dims)
            .contract(kernel.reshape(kernel_dims), contract_dims, output_kernel)
            .reshape(post_contract_dims));
  }      
  
  return Eigen::choose(
        Eigen::Cond<Eigen::internal::traits<Input>::Layout == Eigen::ColMajor>(),
        kernel.reshape(kernel_dims)
            .contract(input
                          .extract_image_patches(
                              kernelRows, kernelCols, row_stride, col_stride,
                              row_in_stride, col_in_stride, padding_type)
                          .reshape(pre_contract_dims),
                      contract_dims, output_kernel)
            .reshape(post_contract_dims),
        input
            .extract_image_patches(kernelRows, kernelCols, row_stride,
                                   col_stride, row_in_stride, col_in_stride,
                                   padding_type)
            .reshape(pre_contract_dims)
            .contract(kernel.reshape(kernel_dims), contract_dims, output_kernel)
            .reshape(post_contract_dims));
 
}



using ColTensor3DFloat = Eigen::Tensor<float, 3, Eigen::ColMajor>;
using ColTensor4DFloat = Eigen::Tensor<float, 4, Eigen::ColMajor>;


// This method uses Eigen to extract the patches and then contraction
inline ColTensor3DFloat eigenConv(const ColTensor3DFloat& input,
                                  const ColTensor4DFloat& kernels) {
  return SpatialConvolution(input, kernels);
}