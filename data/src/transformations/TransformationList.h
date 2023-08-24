#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/Datasets.h>
#include <algorithm>
#include <memory>

namespace thirdai::data {

class TransformationList final : public Transformation {
 public:
  explicit TransformationList(std::vector<TransformationPtr> transformations)
      : _transformations(std::move(transformations)) {}

  static auto make(std::vector<TransformationPtr> transformations) {
    return std::make_shared<TransformationList>(std::move(transformations));
  }

  ColumnMap apply(ColumnMap columns, State& state) const final {
    for (const auto& transformation : _transformations) {
      // This is a shallow copy and not expensive since columns are stored as
      // shared pointers.
      columns = transformation->apply(std::move(columns), state);
    }

    return columns;
  }

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  const auto& transformations() const { return _transformations; }

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<TransformationList> load(const std::string& filename);

  static std::shared_ptr<TransformationList> load_stream(
      std::istream& input_stream);

 private:
  std::vector<TransformationPtr> _transformations;

  // Private constructor for cereal.
  TransformationList(){};

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using TransformationListPtr = std::shared_ptr<TransformationList>;

}  // namespace thirdai::data
