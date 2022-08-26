#pragma once

#include "CompressedVector.h"
#include <cstddef>
#include <cstdint>
#include <random>

namespace thirdai::compression {

// Interface for a DragonVector
template <class T>
class DragonVector final : public CompressedVector<T> {
 public:
  // defining the constructors for the class
  DragonVector<T>() {}

  /*
   * If we are constructing a dragon vector from (indices,values) then we need
   * to know the size of the original vector. Keeping track of the original size
   * is important when we want to decompress a vector.
   */
  DragonVector(const std::vector<T>& vec, float compression_density,
               int seed_for_hashing);

  DragonVector(std::vector<uint32_t> indices, std::vector<T> values,
               uint32_t original_size, int seed_for_hashing);

  DragonVector(const T* values, uint32_t size, float compression_density,
               int seed_for_hashing);

  // using "copy-swap idiom" for = operator. This implementation makes sure that
  // we do not have to check for self-reference.
  DragonVector& operator=(DragonVector<T> vec) {
    swap(*this, vec);
    return *this;
  }

  DragonVector(DragonVector<T>&& vec) : DragonVector<T>() { swap(*this, vec); }

  friend void swap(DragonVector<T>& first, DragonVector<T>& second) {
    std::swap(first._indices, second._indices);
    std::swap(first._values, second._values);
    std::swap(first._min_sketch_size, second._min_sketch_size);
    std::swap(first._original_size, second._original_size);
    std::swap(first._compression_density, second._compression_density);
    std::swap(first._seed_for_hashing, second._seed_for_hashing);
  }

  /*
   * Implementing std::vector's standard methods for the class
   */

  T get(uint32_t index) const final;

  void set(uint32_t index, T value) final;

  // we are only writing for a simple assign now, later expand to iterators and
  // array as well?
  void assign(uint32_t size, T value);

  void assign(uint32_t size, uint32_t index, T value,
              uint32_t original_size = 0);

  void clear() final;

  /*
   * Implementing Operator methods for the class
   */

  /*
   * DragonSketch are by default not additive. Ideally, we would be "adding"
   * dragon sketches by concatenating them. We should still implement + operator
   * because it comes in handy in a distributed setting where we may want to
   * all-reduce dragon vectors.
   */

  DragonVector<T> operator+(DragonVector<T> const& vec) const;

  DragonVector<T>& operator+=(DragonVector<T> const& vec);
  /*
   * To-Do(Shubh):
   * This method should return a reference to the element at the index so that
   * we can do things like vector[i]=a.
   */

  T operator[](uint32_t index) const final;

  /*
   * Implementing utility methods for the class
   */

  void extend(const DragonVector<T>& vec);

  /*
   * Splitting a dragon vector into smaller parts. This is useful when we are
   * training in a distributed setting with ring-all-reduce framework. We need
   * to split the data into smaller parts and communicate. The parameters
   * _original_size, _seed_for_hashing remain the same for the split vectors.
   */

  std::vector<DragonVector<T>> split(size_t number_chunks) const;

  /*
   * Dragon vectors are not additive by default. But we can still define schemes
   * to add them up.
   */

  bool isAdditive() const final;

  std::vector<uint32_t> getIndices() { return _indices; }

  std::vector<T> getValues() { return _values; }

  int getSeedForHashing() const { return _seed_for_hashing; }

  uint32_t getOriginalSize() const { return _original_size; }

  float getCompressionDensity() const { return _compression_density; }

  uint32_t size() const { return static_cast<uint32_t>(_indices.size()); }

  std::string getCompressionScheme() const final;

  static DragonVector<T> addVectors(
      const std::vector<std::unique_ptr<DragonVector<T>>>& vec) {
    DragonVector<T> result_vec = *vec[0].get();
    for (size_t i = 1; i < vec.size(); i++) {
      result_vec = *vec[i].get() + result_vec;
    }
    return result_vec;
  }

  static DragonVector<T> addVectors(
      const std::vector<std::unique_ptr<CompressedVector<T>>>& vec) {
    return addVectors(castToDragonVector(vec));
  }

  static DragonVector<T> concatVectors(
      const std::vector<std::unique_ptr<CompressedVector<T>>>& vec) {
    return concatVectors(castToDragonVector(vec));
  }

  static DragonVector<T> concatVectors(
      const std::vector<std::unique_ptr<DragonVector<T>>>& vec) {
    DragonVector<T> return_vec = *vec[0].get();
    for (size_t i = 1; i < vec.size(); i++) {
      return_vec.extend(*vec[i].get());
    }
    return return_vec;
  }

  static std::vector<std::unique_ptr<DragonVector<T>>> castToDragonVector(
      const std::vector<std::unique_ptr<CompressedVector<T>>>& vec) {
    std::vector<std::unique_ptr<DragonVector<T>>> final_vec;
    final_vec.reserve(vec.size());
    for (const auto& i : vec) {
      final_vec.push_back(std::make_unique<DragonVector<T>>(
          *dynamic_cast<DragonVector<T>*>(i.get())));
    }
    return final_vec;
  }

  /*
   * We are storing indices,values tuple hence, decompressing is just
   * putting corresponding values for the stored indices
   */
  std::vector<T> decompressVector() const final;

 private:
  /*
   * If we add a lot of compression schemes, we should have a sparse vector
   * object rather than indices, values, size parameters. A lot of compression
   * schemes such as topk, randomk, dragon, dgc uses a sparse vector
   */

  std::vector<uint32_t> _indices;
  std::vector<T> _values;
  uint32_t _min_sketch_size = 10;
  uint32_t _original_size = 0;
  float _compression_density = 1;
  int _seed_for_hashing;

  void sketchVector(const T* values, T threshold, uint32_t size,
                    uint32_t sketch_size);
};
}  // namespace thirdai::compression

/*
 * Exception handling?
 * While getting element at a location, check if element is present at the
 * hashed location in the indices array. Size exceptions for the methods. Not
 * adding two vectors with different seed_for_hashing? Should we make this
 * sketch all reducible? or just concatenation? minimum size of the sketch
 */