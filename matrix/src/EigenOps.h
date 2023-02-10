// #include <eigen/unsupported/Eigen/CXX11/Tensor>

#include "wrappers/src/EigenDenseWrapper.h"
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

// This method uses Eigen to extract the patches and then contraction
inline RowTensor3DFloat eigenConv(const RowTensor3DFloat& input,
                                  const RowTensor4DFloat& kernels) {
  int32_t kern_filters = kernels.dimension(0);
  int32_t kern_channels = kernels.dimension(1);
  int32_t kern_width = kernels.dimension(2);
  int32_t kern_height = kernels.dimension(3);

  int32_t image_channels = input.dimension(0);
  int32_t image_width = input.dimension(1);
  int32_t image_height = input.dimension(2);

  if (image_channels != kern_channels) {
    throw std::invalid_argument(
        "Number of channels in input must equal number of channels in kernels "
        "(found " +
        std::to_string(image_channels) + " image channels and " +
        std::to_string(kern_channels) + " kernel channels).");
  }

  int32_t stride_w = 1;
  int32_t stride_h = 1;
  int32_t dilation_w = 1;
  int32_t dilation_h = 1;

  int32_t padding_top = 0;
  int32_t padding_bottom = 0;
  int32_t padding_left = 0;
  int32_t padding_right = 0;

  // Only works for strides = 1
  int32_t output_w =
      image_width + padding_left + padding_right - kern_width + 1;
  int32_t output_h =
      image_height + padding_bottom + padding_top - kern_height + 1;

  Eigen::array<int, 2> pre_contract_dims(
      {output_w * output_h, kern_width * kern_height * kern_channels});
  Eigen::array<int, 2> kernel_dims(
      {kern_channels * kern_width * kern_height, kern_filters});
  Eigen::array<int, 3> post_contract_dims({output_w, output_h, kern_filters});

  Eigen::array<Eigen::IndexPair<int>, 1> normal_matrix_multiplication;
  normal_matrix_multiplication[0] = Eigen::IndexPair<int>(1, 0);

  RowTensor3DFloat outputTensor =
      input
          .extract_image_patches(
              /* patch_rows = */ kern_width, /* patch_cols = */ kern_height,
              /* row_stride = */ stride_w, /* col_stride = */ stride_h,
              /* in_row_stride = */ dilation_w,
              /* in_col_stride = */ dilation_h,
              /*row_inflate_stride = */ 1, /*col_inflate_stride = */ 1,
              /* padding_top = */ padding_top,
              /* padding_bottom = */ padding_bottom,
              /* padding_left = */ padding_left,
              /* padding_right = */ padding_right, /* padding_value = */ 0)
          .reshape(pre_contract_dims)
          .contract(kernels.reshape(kernel_dims), normal_matrix_multiplication)
          .reshape(post_contract_dims);
  return outputTensor;
}

inline RowTensor3DFloat tfEigenConv(
    const RowTensor3DFloat& input, const RowTensor4DFloat& kern,
    const Eigen::Index row_stride = 1, const Eigen::Index col_stride = 1,
    const int32_t padding_type_int = 2, const Eigen::Index row_in_stride = 1,
    const Eigen::Index col_in_stride = 1, Eigen::Index padding_top = 0,
    Eigen::Index padding_bottom = 0, Eigen::Index padding_left = 0,
    Eigen::Index padding_right = 0) {
  Eigen::PaddingType padding_type = padding_type_int == 2
                                        ? Eigen::PaddingType::PADDING_SAME
                                        : Eigen::PaddingType::PADDING_VALID;

  typedef typename Eigen::internal::traits<RowTensor3DFloat>::Index TensorIndex;
  typedef
      typename Eigen::internal::traits<RowTensor3DFloat>::Scalar InputScalar;

  Eigen::TensorRef<Eigen::Tensor<
      InputScalar, Eigen::internal::traits<RowTensor3DFloat>::NumDimensions,
      Eigen::internal::traits<RowTensor3DFloat>::Layout, TensorIndex> >
      in(input);


  // Number of filters to apply. This is the same as the output depth of the
  // result
  const TensorIndex kernelFilters = kern.dimensions()[3];
  // Number of channels. This is the same as the input depth.
  const TensorIndex kernelChannels = kern.dimensions()[2];
  const TensorIndex kernelRows = kern.dimensions()[1];
  const TensorIndex kernelCols = kern.dimensions()[0];

  const Eigen::Index kernelRowsEff =
      kernelRows + (kernelRows - 1) * (row_in_stride - 1);
  const Eigen::Index kernelColsEff =
      kernelCols + (kernelCols - 1) * (col_in_stride - 1);

  Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

  const TensorIndex InputRows = in.dimension(1);
  const TensorIndex InputCols = in.dimension(0);
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
      eigen_assert(false && "unexpected padding");
    }
  }

  // Molds the output of the patch extraction code into a 2d tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  // kernels
  // - the second dimension (dims[1]): everything else
  Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
  pre_contract_dims[1] = kernelChannels * kernelRows * kernelCols;
  pre_contract_dims[0] = out_height * out_width;

  // Molds the output of the contraction into the shape expected by the used
  // (assuming this is ColMajor):
  // - 1st dim: kernel filters
  // - 2nd dim: output height
  // - 3rd dim: output width
  // - 4th dim and beyond: everything else including batch size
  Eigen::DSizes<TensorIndex, 3> post_contract_dims;
  post_contract_dims[2] = kernelFilters;
  post_contract_dims[1] = out_height;
  post_contract_dims[0] = out_width;

  Eigen::DSizes<TensorIndex, 2> kernel_dims;
  kernel_dims[0] = kernelChannels * kernelRows * kernelCols;
  kernel_dims[1] = kernelFilters;
  if (padding_explicit) {
    return input
        .extract_image_patches(kernelRows, kernelCols, row_stride, col_stride,
                               row_in_stride, col_in_stride,
                               /*row_inflate_stride=*/1,
                               /*col_inflate_stride=*/1, padding_top,
                               padding_bottom, padding_left, padding_right,
                               /*padding_value=*/static_cast<InputScalar>(0))
        .reshape(pre_contract_dims)
        .contract(kern.reshape(kernel_dims), contract_dims)
        .reshape(post_contract_dims);
  }
  return input
      .extract_image_patches(kernelRows, kernelCols, row_stride, col_stride,
                             row_in_stride, col_in_stride, padding_type)
      .reshape(pre_contract_dims)
      .contract(kern.reshape(kernel_dims), contract_dims)
      .reshape(post_contract_dims);
}
