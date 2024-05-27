#include "SamplingConfig.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/neuron_index/LshIndex.h>
#include <algorithm>

namespace thirdai::bolt {

NeuronIndexPtr DWTASamplingConfig::getNeuronIndex(uint32_t layer_dim,
                                                  uint32_t input_dim) const {
  auto hash_fn = getHashFunction(input_dim);
  auto hash_table = getHashTable();
  return LshIndex::make(layer_dim, hash_fn, hash_table);
}

hashing::HashFunctionPtr DWTASamplingConfig::getHashFunction(
    uint32_t input_dim) const {
  return std::make_unique<hashing::DWTAHashFunction>(
      /* input_dim= */ input_dim,
      /* hashes_per_table= */ _hashes_per_table,
      /* num_tables= */ _num_tables,
      /* range_pow= */ _range_pow,
      /* binsize=*/_binsize,
      /* permutations=*/_permutes);
}

hashtable::SampledHashTablePtr DWTASamplingConfig::getHashTable() const {
  return std::make_unique<hashtable::SampledHashTable>(
      /* num_tables= */ _num_tables,
      /* reservoir_size= */ _reservoir_size,
      /* range= */ 1 << _range_pow);
}

std::shared_ptr<DWTASamplingConfig> DWTASamplingConfig::newAutotune(
    uint32_t layer_dim, float sparsity) {
  /**
   * Setup:
   * - We have a set of weight vectors, denoted as 'w'.
   * - We also have a query vector, denoted as 'q'.
   * - We aim to find the weight vector that maximizes the inner product with
   * the query vector, i.e., argmax_{w_i} (inner_product(q, w)).
   *
   * It's important to note that the inner product between 'w_i' and 'q' induces
   * an ordering on the set 'w'. Our goal is to approximate this ordering using
   * Discrete Winner Take All (DWTA).
   *
   * The original autotuner continuously increased the number of hashes per
   * table as the layer dimension (|w|) became larger. This strategy is based on
   * the intuition that, when we are retrieving vectors from a large set, a
   * loose similarity definition can lead us to retrieve irrelevant vectors.
   *
   * However, this approach doesn't consider the query vector. For instance, if
   * the dimension of the query vector is small or it contains few non-zeros,
   * approximating the ordering using DWTA could be difficult when the number of
   * hashes_per_table is high. High number of hashes_per_table essentially
   * randomizes the ordering over the set of weight vectors.
   *
   * To address these issues, we conducted several experiments to determine the
   * optimal hyperparameters for DWTA. The experiment setup can be found at this
   * link:
   * https://www.notion.so/Retrieval-of-Neurons-Correctness-of-DWTA-d3072ed2413b4329a1f58db48ace367f#fc44fda03c4a44f7be9f7c9bc8d17a4c
   *
   * The results of the experiments can be found here:
   * https://www.notion.so/a78a1cb7aeb44c11b50fc640051b5035?v=584d411f646b42c484ab7aae7279df22
   *
   * Additionally, the updated autotuner has been benchmarked against various
   * datasets. The benchmarking results are available here:
   * https://www.notion.so/Retrieval-of-Neurons-Correctness-of-DWTA-d3072ed2413b4329a1f58db48ace367f?pvs=4#fc44fda03c4a44f7be9f7c9bc8d17a4c
   */

  uint32_t hashes_per_table = 1;
  uint32_t sparse_dim = (layer_dim * sparsity);

  uint32_t expected_num_elements_per_bucket;

  /**
   * This is a "magic function". When running grid search, we used some fixed
   * values for expected number of values per bucket. This function is just an
   * extrapolation over those values.
   * While this is not ideal, until we do a grid search on this parameter or
   * come up with some other formulation, this is what we have.
   */
  expected_num_elements_per_bucket = static_cast<uint32_t>(std::max(
      std::log(layer_dim) * 2 * (layer_dim / (layer_dim + 5000.0)), 1.0));

  uint32_t range_pow = static_cast<uint32_t>(std::max(
      std::floor(std::log2(layer_dim / expected_num_elements_per_bucket)),
      1.0));

  uint32_t binsize = static_cast<uint32_t>(std::floor(
      std::pow(2, static_cast<float>(range_pow) / hashes_per_table)));

  uint32_t num_tables = static_cast<uint32_t>(static_cast<float>(sparse_dim) /
                                              expected_num_elements_per_bucket);

  return std::make_shared<DWTASamplingConfig>(
      /* num_tables= */ num_tables,
      /* hashes_per_table= */ hashes_per_table,
      /* range_pow=*/range_pow,
      /* binsize=*/binsize,
      /* reservoir_size= */ 4 * expected_num_elements_per_bucket,
      /* permutations=*/2);
}

std::shared_ptr<DWTASamplingConfig> DWTASamplingConfig::oldAutotune(
    uint32_t layer_dim, float sparsity) {
  // The number of items in the table is equal to the number of neurons in
  // this layer, which is stored in the "dim" variable. By analyzing the
  // hash table, we find that
  // E(num_elements_per_bucket) = dim / 2^(range_pow) = sparsity * dim *
  // safety_factor / num_tables The first expression comes from analyzing a
  // single hash table, while the second comes from analyzing the total number
  // of elements returned across the tables. safety_factor is a constant that
  // equals how many more times elements we want to expect to have across
  // tables than the minimum. Simplifying, we have 1 / 2^(range_pow) =
  // sparsity * safety_factor / num_tables This leaves us with 3 free
  // variables: safety_factor, num_tables, and hashes_per_table.

  // First, we will set num_tables_guess = 128 and safety_factor = 1.
  // num_tables_guess is an initial guess to get a good value for range_pow,
  // but we do not find the final num_tables until below because the rounding
  // in the range_pow calculation step can mess things up.
  uint32_t num_tables_guess = 128;
  float safety_factor = 1;

  // We can now set range_pow: manipulating the equation, we have that
  // range_pow = log2(num_tables / (sparsity * safety_factor))
  float range_pow_float =
      std::log2(num_tables_guess / (sparsity * safety_factor));
  // By the properties of DWTA, hashes_per_table = range_pow / 3.
  float hashes_per_table_float = range_pow_float / 3;
  // We now round hashes_per_table to the nearest integer.
  // Using round is more accurate than truncating it down.
  uint32_t hashes_per_table = std::round(hashes_per_table_float);
  // Finally, hashes_per_table needs to be clipped, and then we can
  // recalculate range_pow
  hashes_per_table =
      std::clamp<uint32_t>(hashes_per_table, /* low = */ 2, /* high = */ 8);
  uint32_t range_pow = hashes_per_table * 3;

  // We now calculate an exact value for num_tables using the formula
  // num_tables = sparsity * safety_factor * 2^(range_pow)
  uint32_t num_tables = std::round(sparsity * safety_factor * (1 << range_pow));

  // Finally, we want to set reservoir_size to be somewhat larger than
  // the number of expected elements per bucket. Here, we choose as a
  // heuristic 4 times the number of expected elements per bucket. We take
  // a max with 1 to ensure that the reservoir size isn't 0.
  uint32_t expected_num_elements_per_bucket =
      std::max<uint32_t>(layer_dim / (1 << range_pow), 1);
  uint32_t reservoir_size = 4 * expected_num_elements_per_bucket;

  return std::make_shared<DWTASamplingConfig>(
      /* num_tables= */ num_tables,
      /* hashes_per_table= */ hashes_per_table,
      /* range_pow=*/range_pow,
      /* binsize=*/8,
      /* reservoir_size= */ reservoir_size,
      /* permutations=*/std::nullopt);
}

std::shared_ptr<DWTASamplingConfig> DWTASamplingConfig::autotune(
    uint32_t layer_dim, float sparsity, bool experimental_autotune) {
  if (sparsity == 1.0) {
    // If the layer is dense then we don't need to create a sampling config
    // for it.
    return nullptr;
  }

  if (experimental_autotune) {  // NOLINT
    return newAutotune(layer_dim, sparsity);
  }

  return oldAutotune(layer_dim, sparsity);
}

size_t DWTASamplingConfig::estimateHashTableSize(uint32_t dim, float sparsity) {
  auto config = autotune(dim, sparsity, false);

  return config->_num_tables * config->_reservoir_size *
         (1 << config->_range_pow) * 4;
}

NeuronIndexPtr FastSRPSamplingConfig::getNeuronIndex(uint32_t layer_dim,
                                                     uint32_t input_dim) const {
  auto hash_fn = getHashFunction(input_dim);
  auto hash_table = getHashTable();
  return LshIndex::make(layer_dim, hash_fn, hash_table);
}

hashing::HashFunctionPtr FastSRPSamplingConfig::getHashFunction(
    uint32_t input_dim) const {
  return std::make_unique<hashing::FastSRP>(
      /* input_dim= */ input_dim, /* hashes_per_table= */ _hashes_per_table,
      /* num_tables= */ _num_tables);
}

hashtable::SampledHashTablePtr FastSRPSamplingConfig::getHashTable() const {
  return std::make_unique<hashtable::SampledHashTable>(
      /* num_tables= */ _num_tables,
      /* reservoir_size= */ _reservoir_size,
      /* range= */ 1 << _hashes_per_table);
}

template <class Archive>
void SamplingConfig::serialize(Archive& archive) {
  (void)archive;
}

template <class Archive>
void DWTASamplingConfig::serialize(Archive& archive) {
  archive(cereal::base_class<SamplingConfig>(this), _num_tables,
          _hashes_per_table, _range_pow, _binsize, _reservoir_size, _permutes);
}

template <class Archive>
void FastSRPSamplingConfig::serialize(Archive& archive) {
  archive(cereal::base_class<SamplingConfig>(this), _num_tables,
          _hashes_per_table, _reservoir_size);
}

template <class Archive>
void RandomSamplingConfig::serialize(Archive& archive) {
  archive(cereal::base_class<SamplingConfig>(this));
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::DWTASamplingConfig)
CEREAL_REGISTER_TYPE(thirdai::bolt::FastSRPSamplingConfig)
CEREAL_REGISTER_TYPE(thirdai::bolt::RandomSamplingConfig)
