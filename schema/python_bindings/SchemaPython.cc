#include "SchemaPython.h"
#include <pybind11/cast.h>
#include <schema/DynamicCounts.h>
#include <schema/FeatureHashing.h>
#include <schema/Number.h>
#include <schema/DateFeatures.h>
#include <schema/OneHotEncoding.h>
#include <schema/Schema.h>
#include <schema/Text.h>

namespace thirdai::schema::python {

void createSchemaSubmodule(py::module_& module) {
    auto schema_submodule = module.def_submodule("schema");

    py::class_<ABlockConfig, std::shared_ptr<ABlockConfig>> _schema_block_config_(schema_submodule, "BlockConfig",
        "An interface for schema blocks.");
    py::class_<DynamicCountsBlock::DynamicCountsBlockConfig, 
              std::shared_ptr<DynamicCountsBlock::DynamicCountsBlockConfig>, 
              ABlockConfig> _dynamic_counts_block_config_(schema_submodule, "DynamicCountsBlockConfig",
        "An builder for DynamicCountsBlock.");
    py::class_<FeatureHashingBlock::FeatureHashingBlockConfig, 
              std::shared_ptr<FeatureHashingBlock::FeatureHashingBlockConfig>, 
              ABlockConfig> _feature_hashing_block_config_(schema_submodule, "FeatureHashingBlockConfig",
        "An builder for FeatureHashingBlock.");
    py::class_<OneHotEncodingBlock::OneHotEncodingBlockConfig, 
              std::shared_ptr<OneHotEncodingBlock::OneHotEncodingBlockConfig>, 
              ABlockConfig> _one_hot_encoding_block_config_(schema_submodule, "OneHotEncodingBlockConfig",
        "An builder for OneHotEncodingBlock.");
    py::class_<NumberBlock::NumberBlockConfig, 
              std::shared_ptr<NumberBlock::NumberBlockConfig>, 
              ABlockConfig> _number_block_config_(schema_submodule, "NumberBlockConfig",
        "An builder for NumberBlock.");
    py::class_<DateBlock::DateBlockConfig, 
              std::shared_ptr<DateBlock::DateBlockConfig>, 
              ABlockConfig> _date_block_config_(schema_submodule, "DateBlockConfig",
        "An builder for DateBlock.");
    py::class_<CharacterNGramBlock::CharacterNGramBlockConfig, 
              std::shared_ptr<CharacterNGramBlock::CharacterNGramBlockConfig>, 
              ABlockConfig> _character_n_gram_block_config_(schema_submodule, "CharacterNGramBlockConfig",
        "An builder for CharacterNGramBlock.");
    py::class_<WordNGramBlock::WordNGramBlockConfig, 
              std::shared_ptr<WordNGramBlock::WordNGramBlockConfig>, 
              ABlockConfig> _word_n_gram_block_config_(schema_submodule, "WordNGramBlockConfig",
        "An builder for WordNGramBlock.");


    py::class_<DataLoader>(schema_submodule, "DataLoader",
        "A data loader that processes data according to the provided schema.")
        .def(py::init<std::vector<std::shared_ptr<ABlockConfig>>&, 
            std::vector<std::shared_ptr<ABlockConfig>>&, uint32_t>(),
            py::arg("input_block_configs"), py::arg("label_block_configs"), py::arg("batch_size"))
        .def("input_dim", &DataLoader::inputDim)
        .def("label_dim", &DataLoader::labelDim)
        .def("read_csv", &DataLoader::readCSV,
            py::arg("filename"), py::arg("delimiter"))
        .def("export_dataset", &DataLoader::exportDataset,
            py::arg("shuffle") = true);


    py::class_<Window> (schema_submodule, "Window",
        "Window configuration for dynamic count features.")
        .def(py::init<uint32_t, uint32_t>(),
            py::arg("lag"), py::arg("size"));
    
    
    schema_submodule.def("DynamicCounts", &DynamicCountsBlock::Config,
                          py::arg("id_col"), py::arg("timestamp_col"),
                          py::arg("target_col"), py::arg("window_configs"),
                          py::arg("timestamp_fmt"));
    
    schema_submodule.def("FeatureHashing", &FeatureHashingBlock::Config,
                          py::arg("col"), py::arg("n_hashes"),
                          py::arg("out_dim"));
    
    schema_submodule.def("OneHotEncoding", &OneHotEncodingBlock::Config,
                          py::arg("col"), py::arg("out_dim"));

    schema_submodule.def("Number", &NumberBlock::Config,
                          py::arg("col"));

    schema_submodule.def("CharacterNGram", &CharacterNGramBlock::Config,
                          py::arg("col"), py::arg("k"), py::arg("out_dim"));
    
    schema_submodule.def("WordNGram", &WordNGramBlock::Config,
                          py::arg("col"), py::arg("k"), py::arg("out_dim"));

    schema_submodule.def("Date", &DateBlock::Config, 
                          py::arg("col"), py::arg("timestamp_fmt"), py::arg("n_years")=10);
}
}  // namespace thirdai::schema::python