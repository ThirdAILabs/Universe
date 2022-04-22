#include "SchemaPython.h"
#include <pybind11/cast.h>
#include <schema/FeatureHashing.h>
#include <schema/Number.h>
#include <schema/DateFeatures.h>
#include <schema/NumericalLabel.h>
#include <schema/OneHotEncoding.h>
#include <schema/Schema.h>
#include <schema/Text.h>

namespace thirdai::schema::python {

void createSchemaSubmodule(py::module_& module) {
  auto schema_submodule = module.def_submodule("schema");

  py::class_<ABlockBuilder, std::shared_ptr<ABlockBuilder>> _schema_block_builder_(schema_submodule, "BlockBuilder",
      "An interface for schema blocks.");
  py::class_<DynamicCountsBlock::DynamicCountsBlockBuilder, 
             std::shared_ptr<DynamicCountsBlock::DynamicCountsBlockBuilder>, 
             ABlockBuilder> _dynamic_counts_block_builder_(schema_submodule, "DynamicCountsBlockBuilder",
      "An builder for DynamicCountsBlock.");
  py::class_<FeatureHashingBlock::FeatureHashingBlockBuilder, 
             std::shared_ptr<FeatureHashingBlock::FeatureHashingBlockBuilder>, 
             ABlockBuilder> _feature_hashing_block_builder_(schema_submodule, "FeatureHashingBlockBuilder",
      "An builder for FeatureHashingBlock.");
  py::class_<OneHotEncodingBlock::OneHotEncodingBlockBuilder, 
             std::shared_ptr<OneHotEncodingBlock::OneHotEncodingBlockBuilder>, 
             ABlockBuilder> _one_hot_encoding_block_builder_(schema_submodule, "OneHotEncodingBlockBuilder",
      "An builder for OneHotEncodingBlock.");
  py::class_<NumberBlock::NumberBlockBuilder, 
             std::shared_ptr<NumberBlock::NumberBlockBuilder>, 
             ABlockBuilder> _number_block_builder_(schema_submodule, "NumberBlockBuilder",
      "An builder for NumberBlock.");
  py::class_<DateBlock::DateBlockBuilder, 
             std::shared_ptr<DateBlock::DateBlockBuilder>, 
             ABlockBuilder> _date_block_builder_(schema_submodule, "DateBlockBuilder",
      "An builder for DateBlock.");
  py::class_<NumericalLabelBlock::NumericalLabelBlockBuilder, 
             std::shared_ptr<NumericalLabelBlock::NumericalLabelBlockBuilder>, 
             ABlockBuilder> _numerical_label_block_builder_(schema_submodule, "NumericalLabelBlockBuilder",
      "An builder for NumericalLabelBlock.");
  py::class_<CharacterNGramBlock::CharacterNGramBlockBuilder, 
             std::shared_ptr<CharacterNGramBlock::CharacterNGramBlockBuilder>, 
             ABlockBuilder> _character_n_gram_block_builder_(schema_submodule, "CharacterNGramBlockBuilder",
      "An builder for CharacterNGramBlock.");
  py::class_<WordNGramBlock::WordNGramBlockBuilder, 
             std::shared_ptr<WordNGramBlock::WordNGramBlockBuilder>, 
             ABlockBuilder> _word_n_gram_block_builder_(schema_submodule, "WordNGramBlockBuilder",
      "An builder for WordNGramBlock.");



  py::class_<DataLoader>(schema_submodule, "DataLoader",
      "A data loader that processes data according to the provided schema.")
      .def(py::init<std::vector<std::shared_ptr<ABlockBuilder>>&, uint32_t>(),
           py::arg("schema"), py::arg("batch_size"))
      .def("input_feat_dim", &DataLoader::inputFeatDim)
      .def("read_csv", &DataLoader::readCSV,
           py::arg("filename"), py::arg("delimiter"))
      .def("export_dataset", &DataLoader::exportDataset);


  py::class_<Window> (schema_submodule, "Window",
      "Window configuration for dynamic count features.")
      .def(py::init<uint32_t, uint32_t>(),
           py::arg("lag"), py::arg("size"));
  
  
  schema_submodule.def("DynamicCountsBlock", &DynamicCountsBlock::Builder,
                        py::arg("id_col"), py::arg("timestamp_col"),
                        py::arg("target_col"), py::arg("input_window_configs"),
                        py::arg("label_window_configs"), py::arg("timestamp_fmt"));
  
  schema_submodule.def("FeatureHashingBlock", &FeatureHashingBlock::Builder,
                        py::arg("col"), py::arg("n_hashes"),
                        py::arg("out_dim"));
  
  schema_submodule.def("OneHotEncodingBlock", &OneHotEncodingBlock::Builder,
                        py::arg("col"), py::arg("out_dim"));

  schema_submodule.def("NumberBlock", &NumberBlock::Builder,
                        py::arg("col"));

  schema_submodule.def("CharacterNGramBlock", &CharacterNGramBlock::Builder,
                        py::arg("col"), py::arg("k"), py::arg("out_dim"));
  
  schema_submodule.def("WordNGramBlock", &WordNGramBlock::Builder,
                        py::arg("col"), py::arg("k"), py::arg("out_dim"));

  schema_submodule.def("NumericalLabelBlock", &NumericalLabelBlock::Builder, 
                        py::arg("col"));

  schema_submodule.def("DateBlock", &DateBlock::Builder, 
                        py::arg("col"), py::arg("timestamp_fmt"), py::arg("n_years")=10);
}
}  // namespace thirdai::schema::python