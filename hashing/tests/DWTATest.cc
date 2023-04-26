#include <hashing/src/DWTA.h>
#include <hashing/tests/SparseVector.h>
#include <gtest/gtest.h>
#include <compression/src/CompressionUtils.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <unordered_map>

namespace thirdai::hashing {

struct HashParams {
  uint32_t num_tables;
  uint32_t permutations;
  uint32_t hashes_per_table;
};

std::vector<float> noisify_vector(std::vector<float> vec, float range,
                                  float div, float noise_level, uint32_t seed);
using SparseVec = std::pair<std::vector<float>, std::vector<uint32_t>>;
using HashMatrix = std::vector<std::vector<uint32_t>>;
using Matrix = std::vector<std::vector<float>>;

SparseVec sparsify_vector(const std::vector<float>& dense_vec,
                          float sparsity_level) {
  std::vector<float> copy_dense_vec(dense_vec);

  uint32_t top_k = static_cast<uint32_t>(sparsity_level * dense_vec.size());

  std::nth_element(copy_dense_vec.begin(),
                   copy_dense_vec.begin() + copy_dense_vec.size() - top_k,
                   copy_dense_vec.end());

  float estimated_threshold = copy_dense_vec[copy_dense_vec.size() - top_k];

  SparseVec vec;
  for (int i = 0; i < dense_vec.size(); i++) {
    if (vec.first.size() >= top_k) {
      break;
    }
    if ((dense_vec[i]) >= estimated_threshold) {
      vec.first.push_back(dense_vec[i]);
      vec.second.push_back(i);
    }
  }
  return vec;
}

std::vector<float> makeDenseVec(uint32_t size, int range, float div,
                                uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<float> vec(size, 0);

  std::uniform_int_distribution<int> dist(-range, range);
  // #pragma omp parallel for default(none) shared(vec, size, dist, rng, div)
  for (uint32_t i = 0; i < size; i++) {
    vec[i] = static_cast<float>(dist(rng) / div);
  }
  return vec;
}

TEST(DWTATest, TestSparseDenseOverlap) {
  uint32_t size = 100000;

  float div = 64;
  float range = 1024;

  float sparsity_level = 0.2;
  std::vector<float> dense_vec = makeDenseVec(size, range, div, 0);

  SparseVec sparse_vec = sparsify_vector(dense_vec, sparsity_level);

  uint32_t num_tables = 51, hashes_per_table = 4;
  DWTAHashFunction hash(
      /* input_dim= */ size, /* _hashes_per_table= */ hashes_per_table,
      /* _num_tables= */ num_tables, /* range_pow= */ 3 * hashes_per_table,
      /*permutes=*/8,
      /* seed= */ 59302);

  std::vector<uint32_t> dense_output_hashes(num_tables),
      sparse_output_hashes(num_tables);

  hash.hashSingleDense(dense_vec.data(), dense_vec.size(),
                       dense_output_hashes.data());

  hash.hashSingleSparse(sparse_vec.second.data(), sparse_vec.first.data(),
                        sparse_vec.second.size(), sparse_output_hashes.data());

  uint32_t overlaps = 0;
  for (int i = 0; i < num_tables; i++) {
    if (sparse_output_hashes[i] == dense_output_hashes[i]) {
      overlaps++;
    }
  }

  std::cout << "overlaps: " << overlaps
            << " ratio: " << static_cast<float>(overlaps) / num_tables
            << std::endl;
}

template <class T>
void print_vecs(const std::vector<T>& vec) {
  for (const auto& x : vec) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
}

template <typename S, typename T>
void print_pair_vec(const std::vector<std::pair<S, T>>& vec) {
  for (const auto& x : vec) {
    std::cout << x.first << " " << x.second << std::endl;
  }
}

void print_mat(const Matrix& mat) {
  for (const auto& x : mat) {
    print_vecs(x);
  }
}

Matrix generate_weight_matrix(uint32_t num_vectors, uint32_t dim, float range,
                              float div) {
  Matrix mat;
  mat.resize(num_vectors);
  for (uint32_t i = 0; i < num_vectors; i++) {
    mat[i] = makeDenseVec(dim, range, div, /*seed=*/i);
  }
  return mat;
}

Matrix generate_vectors_for_matrix(Matrix weights, float range, float div,
                                   float noise_level) {
  Matrix vecs;
  vecs.resize(weights.size());
  if (noise_level == 0) {
    return weights;
  }
  srand(0);
  for (int i = 0; i < weights.size(); i++) {
    vecs[i] = noisify_vector(weights[i], range, div, noise_level, 100 * rand());
  }

  return vecs;
}

Matrix convert_dense_matrix_to_sparse(const Matrix& mat, float sparsity_level,
                                      bool one_hot) {
  Matrix sparse_vectors;
  sparse_vectors.resize(mat.size());

#pragma omp parallel for default(none) \
    shared(mat, sparsity_level, one_hot, sparse_vectors)
  for (int i = 0; i < mat.size(); i++) {
    const auto& x = mat[i];
    auto sparse_vec = sparsify_vector(x, sparsity_level);
    std::vector<float> temp(x.size(), 0);
    for (int i = 0; i < sparse_vec.first.size(); i++) {
      if (one_hot) {
        temp[sparse_vec.second[i]] = 1;
        // if (rand() % 3 != 0) {
        //   continue;
        // }
        // temp[sparse_vec.second[i]] = 1 + sparse_vec.first[i] / 10000;
      } else {
        temp[sparse_vec.second[i]] = sparse_vec.first[i];
      }
    }
    sparse_vectors[i] = temp;
  }
  return sparse_vectors;
}

std::vector<float> noisify_vector(std::vector<float> vec, float range,
                                  float div, float noise_level, uint32_t seed) {
  float normalised_div = div / noise_level;
  std::vector<float> noise =
      makeDenseVec(vec.size(), range, normalised_div, seed);

  for (size_t i = 0; i < vec.size(); i++) {
    if (noise_level < 1) {
      noise[i] += vec[i] * (1 / (1 + 2 * noise_level));
    }
  }
  return noise;
}

// Sorting function
template <class T>
std::vector<std::pair<T, uint32_t>> sort_vec(const std::vector<T>& vec) {
  std::vector<std::pair<T, uint32_t>> pair_vec(vec.size());
  for (int i = 0; i < vec.size(); i++) {
    pair_vec[i] = {vec[i], i};
  }

  std::sort(pair_vec.begin(), pair_vec.end(),
            [](auto& left, auto& right) { return left.first > right.first; });

  return pair_vec;
}

float innerproduct(const std::vector<float>& vec1,
                   const std::vector<float>& vec2) {
  float prod = 0;
  // #pragma omp parallel for reduction(+ : prod) default(none) shared(vec1,
  // vec2)
  for (int i = 0; i < vec1.size(); i++) {
    prod += vec1[i] * vec2[i];
  }
  return prod;
}

std::vector<float> innerproduct(const Matrix& mat,
                                const std::vector<float>& vec) {
  std::vector<float> prod;
  for (const auto& x : mat) {
    prod.push_back(innerproduct(x, vec));
  }
  return prod;
}

Matrix innerproduct(const Matrix& mat1, const Matrix& mat2) {
  Matrix prod;
  prod.reserve(mat1.size());
  for (const auto& x : mat2) {
    prod.emplace_back(innerproduct(mat1, x));
  }
  return prod;
}

void normalize(std::vector<float>& vec) {
  float norm = std::sqrt(innerproduct(vec, vec));
  // #pragma omp parallel for default(none) shared(vec, norm)
  for (float& x : vec) {
    x = x / norm;
  }
}

void normalize(Matrix& mat) {
  for (auto& x : mat) {
    normalize(x);
  }
}
Matrix sample_vectors(uint32_t number_samples, const Matrix& matrix) {
  std::vector<std::vector<float>> sampled_vectors;
  uint32_t matrix_size = static_cast<uint32_t>(matrix.size());

  if (number_samples > matrix_size) {
    throw std::runtime_error(
        "number_samples must be less than or equal to the number of vectors in "
        "the matrix.");
  }

  // Create an index vector containing all indices from 0 to matrix_size - 1
  std::vector<uint32_t> indices(matrix_size);
  for (uint32_t i = 0; i < matrix_size; ++i) {
    indices[i] = i;
  }

  // Shuffle the index vector
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  // Select the first number_samples indices from the shuffled index vector
  // and use them to pick the corresponding vectors from the matrix
  sampled_vectors.reserve(number_samples);
  for (uint32_t i = 0; i < number_samples; ++i) {
    sampled_vectors.push_back(matrix[indices[i]]);
  }

  return sampled_vectors;
}

// Sparse hashing a single vector
void hash_sparse_vec(std::vector<uint32_t>& hashes,
                     const std::vector<float>& vec, DWTAHashFunction& hash) {
  assert(sparse_size > 0);

  std::vector<uint32_t> indices;
  std::vector<float> values;

  size_t non_zero_count = 0;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] != 0) {
      indices.push_back(static_cast<uint32_t>(i));
      values.push_back(vec[i]);
      non_zero_count++;
    }
  }

  // Compute hashes
  hash.hashSingleSparse(indices.data(), values.data(), non_zero_count,
                        hashes.data());
}

HashMatrix calculate_hashes(const Matrix& mat, DWTAHashFunction& hash,
                            bool sparse_vec) {
  HashMatrix hash_matrix(mat.size(),
                         std::vector<uint32_t>(hash.numTables(), 0));
#pragma omp parallel for default(none) \
    shared(hash_matrix, hash, mat, sparse_vec)
  for (size_t i = 0; i < mat.size(); i++) {
    if (sparse_vec) {
      hash_sparse_vec(hash_matrix[i], mat[i], hash);
    } else {
      hash.hashSingleDense(mat[i].data(), mat[i].size(), hash_matrix[i].data());
    }
  }
  return hash_matrix;
}

std::vector<uint32_t> get_collisions(const HashMatrix& mat,
                                     const std::vector<float>& vec,
                                     DWTAHashFunction& hash, bool sparse_vec) {
  const size_t mat_size = mat.size();
  const size_t num_tables = hash.numTables();

  std::vector<uint32_t> collisions(mat_size);

  // Compute hashes for the input vector
  std::vector<uint32_t> vec_hashes(num_tables);
  if (sparse_vec) {
    hash_sparse_vec(vec_hashes, vec, hash);
  } else {
    hash.hashSingleDense(vec.data(), vec.size(), vec_hashes.data());
  }

// Calculate collisions
#pragma omp parallel for default(none) \
    shared(mat, mat_size, num_tables, hash, vec_hashes, collisions)
  for (int i = 0; i < mat_size; i++) {
    const auto& row = mat[i];
    uint32_t overlap = 0;
    for (int j = 0; j < num_tables; j++) {
      overlap += (row[j] == vec_hashes[j]);
    }
    collisions[i] = overlap;
  }
  return collisions;
}

float calculate_topk_overlap(const Matrix& mat, const HashMatrix& hash_matrix,
                             const std::vector<float>& vec,
                             DWTAHashFunction& hash, uint32_t topk,
                             bool sparse_vec) {
  // Calculate collisions
  auto collisions = get_collisions(hash_matrix, vec, hash, sparse_vec);

  // Calculate inner products
  const size_t mat_size = mat.size();
  std::vector<float> inner_products(mat_size);
#pragma omp parallel for default(none) \
    shared(mat, vec, inner_products, mat_size)
  for (int i = 0; i < mat_size; i++) {
    inner_products[i] = innerproduct(mat[i], vec);
  }

  // Sort collisions and inner products
  auto sorted_collisions = sort_vec(collisions);
  auto sorted_products = sort_vec(inner_products);

  // Check if the vectors are sorted
  for (int i = 0; i < sorted_collisions.size() - 1; i++) {
    assert(sorted_collisions[i].first >= sorted_collisions[i + 1].first);
    assert(sorted_products[i].first >= sorted_products[i + 1].first);
  }

  // Get top-k collisions and inner products
  std::unordered_set<uint32_t> top_collisions, top_prods, intersection;
  float overlap = 0;
  for (int i = 0; i < topk; i++) {
    top_collisions.insert(sorted_collisions[i].second);
    top_prods.insert(sorted_products[i].second);
  }

  // This code is used when we want to measure the overlap of top-k of both
  // collisions and products
  // for (auto x : top_collisions) {
  //   if (top_prods.find(x) != top_prods.end()) {
  //     intersection.insert(x);
  //   }
  // }

  // Check if the highest scoring product is in the top-k collisions
  return top_collisions.find(sorted_products[0].second) == top_collisions.end()
             ? 0
             : 1;
}

float calculate_topk_overlap(const Matrix& weights, const Matrix& vectors,
                             DWTAHashFunction& hash, uint32_t topk,
                             bool sparse_vec) {
  float overlap = 0;
  HashMatrix hash_matrix = calculate_hashes(weights, hash, sparse_vec);
  for (const auto& x : vectors) {
    overlap +=
        calculate_topk_overlap(weights, hash_matrix, x, hash, topk, sparse_vec);
  }
  return overlap / vectors.size();
}

void run_experiment(
    uint32_t dim, float topk, uint32_t num_vectors, float noise_level,
    bool use_sparse_vectors, bool one_hot, float sparsity_level,
    HashParams params,
    std::optional<uint32_t> number_query_vectors = std::nullopt) {
  float div = 32;
  float range = 128;

  uint32_t num_tables = params.num_tables,
           hashes_per_table = params.hashes_per_table;
  DWTAHashFunction hash(
      /* input_dim= */ dim, /* _hashes_per_table= */ hashes_per_table,
      /* _num_tables= */ num_tables, /* range_pow= */ 3 * hashes_per_table,
      /*permutes=*/params.permutations,
      /* seed= */ 59302);

  auto weights = generate_weight_matrix(num_vectors, dim, range, div);
  auto vectors = generate_vectors_for_matrix(weights, range, div, noise_level);

  print_mat(weights);
  print_mat(vectors);
  if (number_query_vectors != std::nullopt) {
    vectors = sample_vectors(number_query_vectors.value(), vectors);
  }
  print_mat(vectors);

  uint32_t sparse_size = static_cast<uint32_t>(sparsity_level * dim);

  normalize(weights);

  if (use_sparse_vectors && sparsity_level < 1) {
    vectors = convert_dense_matrix_to_sparse(vectors, sparsity_level, one_hot);
  }
  normalize(vectors);

  print_mat(weights);
  print_mat(vectors);

  // print_mat(innerproduct(weights, vectors));

  // print_vecs(get_collisions(weights, vectors[0], hash));
  // print_vecs(innerproduct(weights, vectors[0]));
  // print_pair_vec(sort_vec(get_collisions(weights, vectors[0], hash)));
  std::cout << "overlap: "
            << calculate_topk_overlap(
                   weights, vectors, hash, topk,
                   (use_sparse_vectors && sparsity_level < 1))
            << std::endl;
}

TEST(DWTATest, trial) {
  uint32_t dim = 10;
  uint32_t topk = 10;
  uint32_t num_vectors = 5;
  float noise_level = 0.1;
  float sparsity_level = 0.1;

  HashParams param = HashParams({10, 1, 1});
  run_experiment(
      /*dim=*/dim,
      /*topk=*/topk,
      /*num_vectors=*/num_vectors,
      /*noise_level=*/noise_level,
      /*use_sparse_vectors=*/false,
      /*one_hot=*/false,
      /*sparsity_level=*/sparsity_level,
      /*params=*/param);
}

TEST(DWTATest, runner) {
  uint32_t dim = 100'00;
  uint32_t topk = 10;
  uint32_t num_vectors = 100;
  float noise_level = 0.1;

  std::vector<float> sparsity_levels({0.01, 0.1, 0.5, 1});

  auto start_time = std::chrono::high_resolution_clock::now();
  for (auto x : sparsity_levels) {
    std::vector<HashParams> params;
    std::vector<uint32_t> num_tables({51, 121, 171, 251});
    std::vector<uint32_t> permutations({1, 2, 4, 8});
    std::vector<uint32_t> hashes_per_table({1, 2, 4, 8});

    for (auto a : num_tables) {
      for (auto b : permutations) {
        for (auto c : hashes_per_table) {
          params.emplace_back(HashParams({a, b, c}));
        }
      }
    }
    int i = 0;
    for (auto param : params) {
      std::cout << "Params: "
                << "sparsity: " << x << " num_tables: " << param.num_tables
                << " permutations: " << param.permutations
                << " hashes_per_table: " << param.hashes_per_table << "\n ";
      std::cout << "Dense" << std::endl;
      run_experiment(
          /*dim=*/dim,
          /*topk=*/topk,
          /*num_vectors=*/num_vectors,
          /*noise_level=*/noise_level,
          /*use_sparse_vectors=*/false,
          /*one_hot=*/false,
          /*sparsity_level=*/x, param);
      std::cout << "Sparse" << std::endl;
      run_experiment(
          /*dim=*/dim,
          /*topk=*/topk,
          /*num_vectors=*/num_vectors,
          /*noise_level=*/noise_level,
          /*use_sparse_vectors=*/true,
          /*one_hot=*/false,
          /*sparsity_level=*/x, param);
      std::cout << "Sparse One-hot" << std::endl;
      run_experiment(
          /*dim=*/dim,
          /*topk=*/topk,
          /*num_vectors=*/num_vectors,
          /*noise_level=*/noise_level,
          /*use_sparse_vectors=*/true,
          /*one_hot=*/true,
          /*sparsity_level=*/x, param);
    }
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time)
          .count();
  std::cout << "Execution time: " << duration << " seconds" << std::endl;
}
}  // namespace thirdai::hashing
