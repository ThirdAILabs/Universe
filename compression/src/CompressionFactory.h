#pragma once

#include "CompressedVector.h"
#include "CompressionUtils.h"
#include "CountSketch.h"
#include "DragonVector.h"
#include <compression/src/CompressionUtils.h>
#include <cstddef>
#include <stdexcept>
#include <variant>
namespace thirdai::compression {

/*
 * We are using visitor patterns to deal with runtime polymorphism of compressed
 * vector objects. Using visitor pattern also makes sure that we don't have to
 * change the object implementations whenever we want to add a new feature.
 * For example, we have a binary operation called extend that operates on two
 * compressed vectors of the same type. Adding two compressed vector of
 * different classes should throw an error. Using visitor pattern, we can handle
 * these exceptions without modifying the existing classes.
 */
template <class T>
class ExtendVisitor {
 public:
  void operator()(DragonVector<T>& vector_to_extend,
                  const DragonVector<T>& vector_to_extend_with) {
    vector_to_extend.extend(vector_to_extend_with);
  }
  void operator()(CountSketch<T>& vector_to_extend,
                  const CountSketch<T>& vector_to_extend_with) {
    vector_to_extend.extend(vector_to_extend_with);
  }
  void operator()(DragonVector<T>& vector_to_extend,
                  const CountSketch<T>& vector_to_extend_with) {
    (void)vector_to_extend;
    (void)vector_to_extend_with;
    throw std::invalid_argument(
        "Cannot extend a DragonVector with a CountSketch");
  }
  void operator()(CountSketch<T>& vector_to_extend,
                  const DragonVector<T>& vector_to_extend_with) {
    (void)vector_to_extend;
    (void)vector_to_extend_with;
    throw std::invalid_argument(
        "Cannot extend a CountSketch with a DragonVector");
  }
};

template <class T>
class AddVisitor {
 public:
  void operator()(DragonVector<T>& vector_to_add_to,
                  const DragonVector<T>& vector_to_add) {
    vector_to_add_to.add(vector_to_add);
  }
  void operator()(CountSketch<T>& vector_to_add_to,
                  const CountSketch<T>& vector_to_add) {
    vector_to_add_to.add(vector_to_add);
  }
  void operator()(DragonVector<T>& vector_to_add_to,
                  const CountSketch<T>& vector_to_add) {
    (void)vector_to_add_to;
    (void)vector_to_add;
    throw std::invalid_argument("Cannot add a CountSketch to a DragonVector");
  }
  void operator()(CountSketch<T>& vector_to_add_to,
                  const DragonVector<T>& vector_to_add) {
    (void)vector_to_add_to;
    (void)vector_to_add;
    throw std::invalid_argument("Cannot add a DragonVector to a CountSketch");
  }
};

template <class T>
class DecompressVisitor {
 public:
  std::vector<T> operator()(const DragonVector<T>& dragon_vector) {
    return dragon_vector.decompress();
  }
  std::vector<T> operator()(const CountSketch<T>& count_sketch) {
    return count_sketch.decompress();
  }
};

template <class T>
class SizeVisitor {
 public:
  uint32_t operator()(const DragonVector<T>& dragon_vector) {
    return dragon_vector.serialized_size();
  }
  uint32_t operator()(const CountSketch<T>& count_sketch) {
    return count_sketch.serialized_size();
  }
};

template <class T>
class SerializeVisitor {
 public:
  explicit SerializeVisitor<T>(char* pointer_to_write_to)
      : _serialized_data(pointer_to_write_to) {}

  void operator()(const DragonVector<T>& dragon_vector) {
    dragon_vector.serialize(_serialized_data);
  }
  void operator()(const CountSketch<T>& count_sketch) {
    count_sketch.serialize(_serialized_data);
  }

 private:
  char* _serialized_data;
};

template <class T>
class DivideVisitor {
 public:
  explicit DivideVisitor(uint32_t divisor) : _divisor(divisor) {}

  void operator()(DragonVector<T>& dragon_vector) { (void)dragon_vector; }

  void operator()(CountSketch<T>& count_sketch) {
    count_sketch.divide(_divisor);
  }

 private:
  uint32_t _divisor;
};

template <class T>
std::variant<DragonVector<T>, CountSketch<T>> compress(
    const T* values, uint32_t size, const std::string& compression_scheme,
    float compression_density, uint32_t seed_for_hashing,
    uint32_t sample_population_size);

template <class T>
std::variant<DragonVector<T>, CountSketch<T>> concat(
    std::vector<std::variant<DragonVector<T>, CountSketch<T>>>
        compressed_vectors);

template <class T>
std::variant<DragonVector<T>, CountSketch<T>> add(
    std::vector<std::variant<DragonVector<T>, CountSketch<T>>>
        compressed_vectors);
}  // namespace thirdai::compression
