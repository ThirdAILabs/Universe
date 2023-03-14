#include <hashing/src/DWTA.h>
#include <hashing/tests/SparseVector.h>
#include <gtest/gtest.h>
#include <compression/src/CompressionUtils.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

namespace thirdai::hashing {

std::vector<float> noisify_vector(std::vector<float> vec, float range,
                                  float div, float noise_level, uint32_t seed);
using SparseVec = std::pair<std::vector<float>, std::vector<uint32_t>>;
using Matrix = std::vector<std::vector<float>>;

SparseVec sparsify_vector(const std::vector<float>& dense_vec,
                          float sparsity_level) {
  std::vector<float> copy_dense_vec(dense_vec);

  uint32_t top_k = static_cast<uint32_t>(sparsity_level * dense_vec.size());
  float threshold = thirdai::compression::estimateTopKThreshold(
      dense_vec.data(), dense_vec.size(), sparsity_level, 0,
      /*sample_population_size=*/dense_vec.size());

  std::nth_element(copy_dense_vec.begin(),
                   copy_dense_vec.begin() + copy_dense_vec.size() - top_k - 1,
                   copy_dense_vec.end());

  float estimated_threshold = copy_dense_vec[copy_dense_vec.size() - top_k - 1];

  SparseVec vec;
  for (int i = 0; i < dense_vec.size(); i++) {
    if (vec.first.size() >= top_k) {
      break;
    }
    if ((dense_vec[i]) > estimated_threshold) {
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
  for (uint32_t i = 0; i < size; i++) {
    vec[i] = static_cast<float>(dist(rng) / div);
  }
  return vec;
}

TEST(DWTATest, TestSparseDenseOverlap) {
  uint32_t size = 1000;

  float div = 64;
  float range = 1024;

  float sparsity_level = 0.2;
  std::vector<float> dense_vec = makeDenseVec(size, range, div, 0);

  SparseVec sparse_vec = sparsify_vector(dense_vec, sparsity_level);

  uint32_t num_tables = 51, hashes_per_table = 4;
  DWTAHashFunction hash(
      /* input_dim= */ size, /* _hashes_per_table= */ hashes_per_table,
      /* _num_tables= */ num_tables, /* range_pow= */ 3 * hashes_per_table,
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
  std::cout << "here" << std::endl;
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
  for (uint32_t i = 0; i < num_vectors; i++) {
    mat.emplace_back(makeDenseVec(dim, range, div, /*seed=*/i));
  }
  return mat;
}

Matrix generate_vectors_for_matrix(Matrix weights, float range, float div,
                                   float noise_level) {
  Matrix vecs;
  if (noise_level == 0) {
    return weights;
  }
  srand(0);
  for (auto& weight : weights) {
    vecs.emplace_back(
        noisify_vector(weight, range, div, noise_level, 100 * rand()));
  }

  return vecs;
}

Matrix convert_dense_matrix_to_sparse(const Matrix& mat, float sparsity_level,
                                      bool one_hot) {
  Matrix sparse_vectors;
  for (const auto& x : mat) {
    auto sparse_vec = sparsify_vector(x, sparsity_level);
    std::vector<float> temp(x.size(), 0);
    for (int i = 0; i < sparse_vec.first.size(); i++) {
      if (one_hot) {
        temp[sparse_vec.second[i]] = 1;
      } else {
        temp[sparse_vec.second[i]] = sparse_vec.first[i];
      }
    }
    sparse_vectors.emplace_back(std::move(temp));
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
      noise[i] += vec[i] * (1 / (1 + 100 * noise_level));
    }
  }
  return noise;
}

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

std::vector<uint32_t> get_collisions(const Matrix& mat,
                                     const std::vector<float>& vec,
                                     DWTAHashFunction& hash) {
  std::vector<uint32_t> collisions;

  std::vector<uint32_t> vec_hashes(hash.numTables());
  hash.hashSingleDense(vec.data(), vec.size(), vec_hashes.data());

  for (const auto& x : mat) {
    std::vector<uint32_t> hashes(hash.numTables());
    hash.hashSingleDense(x.data(), x.size(), hashes.data());

    uint32_t overlap = 0;
    for (int i = 0; i < vec_hashes.size(); i++) {
      overlap += (hashes[i] == vec_hashes[i]);
    }
    collisions.push_back(overlap);
  }
  return collisions;
}

float innerproduct(const std::vector<float>& vec1,
                   const std::vector<float>& vec2) {
  float prod = 0;
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
  for (auto& x : vec) {
    x = x / norm;
  }
}

void normalize(Matrix& mat) {
  for (auto& x : mat) {
    normalize(x);
  }
}

float calculate_topk_overlap(const Matrix& mat, const std::vector<float>& vec,
                             DWTAHashFunction& hash, uint32_t topk) {
  auto collisions = get_collisions(mat, vec, hash);
  std::vector<float> innerproducts;
  for (const auto& x : mat) {
    innerproducts.emplace_back(innerproduct(x, vec));
  }

  auto sorted_collisions = sort_vec(collisions);
  auto sorted_products = sort_vec(innerproducts);

  for (int i = 0; i < sorted_collisions.size() - 1; i++) {
    assert(sorted_collisions[i].first >= sorted_collisions[i + 1].first);
    assert(sorted_products[i].first >= sorted_products[i + 1].first);
  }

  std::unordered_set<uint32_t> top_collisions, top_prods, intersection;
  float overlap = 0;
  for (int i = 0; i < topk; i++) {
    top_collisions.insert(sorted_collisions[i].second);
    top_prods.insert(sorted_products[i].second);
  }
  for (auto x : top_collisions) {
    if (top_prods.find(x) != top_prods.end()) {
      intersection.insert(x);
    }
  }

  return static_cast<float>(intersection.size()) / topk;
}

float calculate_topk_overlap(const Matrix& weights, const Matrix& vectors,
                             DWTAHashFunction& hash, uint32_t topk) {
  float overlap = 0;
  for (const auto& x : vectors) {
    overlap += calculate_topk_overlap(weights, x, hash, topk);
  }
  return overlap / vectors.size();
}

void run_experiment(uint32_t dim, float topk, uint32_t num_vectors,
                    float noise_level, bool use_sparse_vectors, bool one_hot,
                    float sparsity_level) {
  float div = 32;
  float range = 128;

  uint32_t num_tables = 51, hashes_per_table = 4;
  DWTAHashFunction hash(
      /* input_dim= */ dim, /* _hashes_per_table= */ hashes_per_table,
      /* _num_tables= */ num_tables, /* range_pow= */ 3 * hashes_per_table,
      /* seed= */ 59302);

  auto weights = generate_weight_matrix(num_vectors, dim, range, div);
  auto vectors = generate_vectors_for_matrix(weights, range, div, noise_level);
  normalize(weights);
  normalize(vectors);

  if (use_sparse_vectors && sparsity_level < 1) {
    vectors = convert_dense_matrix_to_sparse(vectors, sparsity_level, one_hot);
  }

  // print_mat(weights);
  // print_mat(vectors);

  // print_mat(innerproduct(weights, vectors));

  // print_vecs(get_collisions(weights, vectors[0], hash));
  // print_vecs(innerproduct(weights, vectors[0]));
  // print_pair_vec(sort_vec(get_collisions(weights, vectors[0], hash)));
  std::cout << "overlap: "
            << calculate_topk_overlap(weights, vectors, hash, topk)
            << std::endl;
}
TEST(DWTATest, runner) {
  run_experiment(
      /*dim=*/1024,
      /*topk=*/1,
      /*num_vectors=*/500,
      /*noise_level=*/0,
      /*use_sparse_vectors=*/false,
      /*one_hot=*/true,
      /*sparsity_level=*/.2);
  run_experiment(
      /*dim=*/1024,
      /*topk=*/50,
      /*num_vectors=*/500,
      /*noise_level=*/0.3,
      /*use_sparse_vectors=*/false,
      /*one_hot=*/true,
      /*sparsity_level=*/.2);
  run_experiment(
      /*dim=*/1024,
      /*topk=*/50,
      /*num_vectors=*/500,
      /*noise_level=*/0.3,
      /*use_sparse_vectors=*/true,
      /*one_hot=*/false,
      /*sparsity_level=*/.2);
  run_experiment(
      /*dim=*/1024,
      /*topk=*/50,
      /*num_vectors=*/500,
      /*noise_level=*/0.3,
      /*use_sparse_vectors=*/true,
      /*one_hot=*/true,
      /*sparsity_level=*/.2);
  run_experiment(
      /*dim=*/1024,
      /*topk=*/50,
      /*num_vectors=*/500,
      /*noise_level=*/0.3,
      /*use_sparse_vectors=*/true,
      /*one_hot=*/true,
      /*sparsity_level=*/.3);
}
}  // namespace thirdai::hashing
