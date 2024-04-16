#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/WordpieceTokenizer.h>
#include <dataset/src/mach/MachIndex.h>
#include <regex>

namespace thirdai::data {

struct MultiSpladeConfig {
  MultiSpladeConfig(const std::vector<std::string>& model_checkpoints,
                    const std::vector<uint32_t>& mach_index_seeds,
                    const std::string& tokenizer_vocab,
                    std::optional<size_t> n_augmented_tokens,
                    std::optional<float> augmentation_frac,
                    bool filter_tokens = true, size_t batch_size = 4096,
                    bool lowercase = true,
                    std::optional<uint32_t> strong_sample_override = 7);

  std::vector<bolt::ModelPtr> models;
  std::vector<dataset::mach::MachIndex> mach_indices;
  dataset::WordpieceTokenizerPtr tokenizer;
  std::optional<size_t> n_augmented_tokens;
  std::optional<float> augmentation_frac;
  bool filter_tokens;
  size_t batch_size = 4096;
  std::optional<uint32_t> strong_sample_override;

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<MultiSpladeConfig> load_stream(
      std::istream& input_stream);

 private:
  std::vector<std::string> _model_checkpoints;
  std::vector<uint32_t> _mach_index_seeds;
  std::string _tokenizer_vocab;
  bool _lowercase = true;
};

class MultiSpladeAugmentation final : public Transformation {
 public:
  MultiSpladeAugmentation(std::string input_column, std::string output_column,
                          const MultiSpladeConfig& config);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

 private:
  std::string decodeTopTokens(const std::vector<BoltVector>& vec,
                              size_t k) const;
  inline size_t tokensToAdd(size_t seq_len) const {
    if (_n_augmented_tokens) {
      return _n_augmented_tokens.value();
    }
    return seq_len * _augmentation_frac.value();
  }

  std::string _input_column;
  std::string _output_column;
  std::vector<bolt::ModelPtr> _models;
  std::vector<dataset::mach::MachIndex> _mach_indices;
  dataset::WordpieceTokenizerPtr _tokenizer;
  std::optional<size_t> _n_augmented_tokens;
  std::optional<float> _augmentation_frac;
  bool _filter_tokens;
  size_t _batch_size = 4096;

  const std::regex _allowed_tokens = std::regex(R"([a-zA-Z]{3,})");
};
}  // namespace thirdai::data