
#include "Indexer.h"

namespace thirdai::automl::deployment {

static std::shared_ptr<Indexer> buildIndexerFromSerializedConfig(
    const std::string& config_file_name) {
  auto flash_index_config = IndexerConfig::load(config_file_name);

  return Indexer::make(flash_index_config);
}

// TODO(blaise): Combine buildFlashIndex and buildFlashIndexPair into one
// function since they are pretty much indentical
template <typename LABEL_T>
std::shared_ptr<Indexer> Indexer::buildFlashIndex(
    const std::string& file_name) {
  auto data = loadDataInMemory(file_name, 0);
  _flash_index = std::make_unique<Flash<LABEL_T>>(
      _flash_index_config->getHashFunction().get());
  _flash_index->addDataset(*data);

  return shared_from_this();
}


template <typename LABEL_T>
std::shared_ptr<Indexer> Indexer::buildFlashIndexPair(
    const std::string& file_name) {
  auto data = loadDataInMemory(file_name, 1);
  _flash_index = std::make_unique<Flash<LABEL_T>>(
      _flash_index_config->getHashFunction().get());
  _flash_index->addDataset(*data);

  return shared_from_this();
}

template std::vector<std::vector<uint32_t>> Indexer::queryIndexFromFile(const std::string& query_file);


template <typename LABEL_T>
std::vector<std::vector<LABEL_T>> Indexer::queryIndexFromFile(
    const std::string& query_file) {
  auto query_data = loadDataInMemory(query_file, 0);
  std::vector<std::vector<LABEL_T>> query_result = _flash_index->queryBatch(
      /* batch = */ query_data, /* top_k = */ Indexer::TOP_K,
      /* pad_zeros = */ true);
  return query_result;
}


template std::vector<std::vector<uint32_t>> Indexer::querySingle(
    const std::string& query);

template <typename LABEL_T>
std::vector<std::vector<LABEL_T>> Indexer::querySingle(
    const std::string& query) {
  auto featurized_query_vector = featurizeSingleQuery(query);

  std::vector<std::vector<LABEL_T>> query_result = _flash_index->queryBatch(
      /* batch = */ BoltBatch(std::move(featurized_query_vector)),
      /* top_k = */ Indexer::TOP_K, /* pad_zeros = */ false);

  return query_result;
}

std::vector<BoltVector> Indexer::featurizeSingleQuery(
    const std::string& query) const {
  dataset::TextBlockPtr char_trigram_block = dataset::CharKGramTextBlock::make(
      /* col = */ 0, /* k = */ 3, /* dim = */ _dimension_for_encodings);
  dataset::TextBlockPtr char_four_gram = dataset::CharKGramTextBlock::make(
      /* col = */ 0, /* k = */ 4, /* dim = */ _dimension_for_encodings);

  std::vector<std::shared_ptr<dataset::Block>> input_blocks{char_trigram_block,
                                                            char_four_gram};

  auto batch_processor = dataset::GenericBatchProcessor(input_blocks, {});

  BoltVector output_vector;
  std::vector<std::string_view> input_vector{
      std::string_view(query.data(), query.length())};
  if (auto exception =
          batch_processor.makeInputVector(input_vector, output_vector)) {
    std::rethrow_exception(exception);
  }
  return {std::move(output_vector)};
}

dataset::BoltDatasetPtr Indexer::loadDataInMemory(
    const std::string& file_name, uint32_t correct_query_column_index) const {
  dataset::TextBlockPtr char_trigram_block = dataset::CharKGramTextBlock::make(
      /* col = */ correct_query_column_index, /* k = */ 3,
      /* dim = */ _dimension_for_encodings);

  dataset::TextBlockPtr char_four_gram = dataset::CharKGramTextBlock::make(
      /* col = */ correct_query_column_index, /* k = */ 4,
      /* dim = */ _dimension_for_encodings);

  std::vector<std::shared_ptr<dataset::Block>> input_blocks{char_trigram_block,
                                                            char_four_gram};

  auto data_loader = dataset::StreamingGenericDatasetLoader(
      /* filename = */ file_name, /* input_blocks = */ input_blocks,
      /* label_blocks = */ {},
      /* batch_size = */ _flash_index_config->batch_size());

  auto [data, _] = data_loader.loadInMemory();

  return data;
}

}  // namespace thirdai::automl::deployment