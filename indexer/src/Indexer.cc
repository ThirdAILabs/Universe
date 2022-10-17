
#include "Indexer.h"

namespace thirdai::automl::deployment {

static std::shared_ptr<Indexer> buildIndexerFromSerializedConfig(
    const std::string& config_file_name) {
  auto flash_index_config = FlashIndexConfig::load(config_file_name);

  return Indexer::make(flash_index_config);
}

template <typename LABEL_T>
std::shared_ptr<Indexer> Indexer::buildFlashIndex(
    const std::string& file_name) {
  auto data = loadDataInMemory(file_name);
  _flash_index =
      std::make_unique<Flash<LABEL_T>>(*_flash_index_config->getHashFunction());
  _flash_index->addDataset(*data);

  return shared_from_this();
}

template <typename LABEL_T>
std::vector<std::vector<LABEL_T>> Indexer::queryIndex(
    const std::string& query_file) {
  auto query_data = loadDataInMemory(query_file);
  std::vector<std::vector<LABEL_T>> results = _flash_index->queryBatch(
      /* batch = */ query_data, /* top_k = */ 5, /* pad_zeros = */ true);
  return results;
}

dataset::BoltDatasetPtr Indexer::loadDataInMemory(
    const std::string& file_name) const {
  dataset::TextBlockPtr char_trigram_block = dataset::CharKGramTextBlock::make(
      /* col = */ 0, /* k = */ 3,
      /* dim = */ _dimension_for_encodings);

  dataset::TextBlockPtr char_four_gram = dataset::CharKGramTextBlock::make(
      /* col = */ 0, /* k = */ 4,
      /* dim = */ _dimension_for_encodings);

  std::vector<std::shared_ptr<dataset::Block>> input_blocks{char_trigram_block,
                                                            char_four_gram};

  auto data_loader = dataset::StreamingGenericDatasetLoader(
      /* filename = */ file_name, /* input_blocks = */ input_blocks,
      /* label_blocks = */ {}, /* batch_size = */ _batch_size);

  auto [data, _] = data_loader.loadInMemory();

  return data;
}

}  // namespace thirdai::automl::deployment