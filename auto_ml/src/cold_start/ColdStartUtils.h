#pragma once

#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/cold_start/ColdStartDataSource.h>
#include <memory>
#include <stdexcept>

namespace thirdai::automl::cold_start {

class ColdStartMetaData;
using ColdStartMetaDataPtr = std::shared_ptr<ColdStartMetaData>;

class ColdStartMetaData {
  /*
   * ColdStartMetaData consist of the fields which is needed by the distributed,
   * We went for a class implementation rather than a struct implementation due
   * to the need of pickling which requires references to move around, hence
   * class made more sense. If we want to add any new field to
   * ColdStartDataSource, we should do it through ColdStartMetaData. So, those
   * changes can easily be included for coldstart distributed training.
   */

 public:
  ColdStartMetaData(std::optional<char> label_delimiter,
                    std::string label_column_name, bool integer_target)
      : _label_delimiter(label_delimiter),
        _label_column_name(std::move(label_column_name)),
        _integer_target(integer_target) {}

  std::string getLabelColumn() const { return _label_column_name; }

  std::optional<char> getLabelDelimiter() { return _label_delimiter; }

  bool integerTarget() const { return _integer_target; }

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static ColdStartMetaDataPtr load(const std::string& filename);

  static ColdStartMetaDataPtr load_stream(std::istream& input_stream);

 private:
  // private constructor for cereal
  ColdStartMetaData(){};

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  std::optional<char> _label_delimiter;
  std::string _label_column_name;
  bool _integer_target;
};

/*
 * This function implements the preprocessing of training data for Cold-Start
 * PreTraining. We need this preprocessing to make sure there is one source for
 * both serial and distributed pre-processing for cold-start.
 */
/*
 * Note(pratkpranav): In the distributed setting, this particular function runs
 * independently on each of the worker, hence almost any additions should be
 * fine except the additions which involves going through the whole training
 * data for once.
 */
dataset::cold_start::ColdStartDataSourcePtr preprocessColdStartTrainSource(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    data::TabularDatasetFactoryPtr& dataset_factory,
    ColdStartMetaDataPtr& metadata);
}  // namespace thirdai::automl::cold_start