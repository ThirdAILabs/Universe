#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/text_generation/GenerativeModel.h>
#include <gtest/gtest.h>

namespace thirdai::bolt::tests {

class MockBackend final : public GenerativeBackend {
 public:
  bolt::TensorPtr nextTokenProbs(
      std::vector<std::vector<uint32_t>>& prompts,
      std::vector<std::vector<std::vector<uint32_t>>>& tokens) final {
    (void)prompts;
    std::vector<std::vector<float>> transition_matrix = {
        {0.1, 0.6, 0.2, 0.1},
        {0.1, 0.2, 0.4, 0.3},
        {0.24, 0.4, 0.3, 0.05},
        {0.05, 0.1, 0.8, 0.05},
    };

    auto output = bolt::Tensor::dense(tokens.size(), 4);
    for (size_t i = 0; i < tokens[0].size(); i++) {
      const auto& scores = transition_matrix[tokens[0][i].back()];
      std::copy(scores.begin(), scores.end(), output->getVector(i).activations);
    }

    return output;
  }

  metrics::History train(const dataset::DataSourcePtr& train_data,
                         float learning_rate, uint32_t epochs,
                         size_t batch_size,
                         const std::vector<std::string>& train_metrics,
                         const dataset::DataSourcePtr& val_data,
                         const std::vector<std::string>& val_metrics,
                         std::optional<size_t> max_in_memory_batches,
                         const DistributedCommPtr& comm) final {
    (void)train_data;
    (void)learning_rate;
    (void)epochs;
    (void)batch_size, (void)train_metrics;
    (void)val_data;
    (void)val_metrics;
    (void)max_in_memory_batches;
    (void)comm;
    return {};
  }

  ModelPtr getBoltModel() final { return nullptr; }
};

TEST(BeamSearchDecoding, GreedySearch) {
  auto model = GenerativeModel::make(std::make_shared<MockBackend>(),
                                     /* allowed_repeats= */ {},
                                     /* punctuation_tokens= */ {},
                                     /* punctuation_repeat_threshold= */ 0.8);

  auto output = model->generate(/* input_tokens= */ {0}, /* prompt= */ {},
                                /* max_predictions= */ 3,
                                /* beam_width= */ 1);

  // 1 -> 2 -> 1 is the "greedy" best path, but it won't predict 0,1,2 again, so
  // it predicts 0.
  std::vector<uint32_t> expected_output = {1, 2, 3};

  ASSERT_EQ(output, expected_output);
}

TEST(BeamSearchDecoding, AllowRepeats) {
  auto model = GenerativeModel::make(std::make_shared<MockBackend>(),
                                     /* allowed_repeats= */ {1},
                                     /* punctuation_tokens= */ {},
                                     /* punctuation_repeat_threshold= */ 0.8);

  auto output = model->generate(/* input_tokens= */ {0}, /* prompt= */ {},
                                /* n_predictions= */ 3,
                                /* beam_width= */ 1);

  std::vector<uint32_t> expected_output = {1, 2, 1};

  ASSERT_EQ(output, expected_output);
}

TEST(BeamSearchDecoding, NoRepeatPunctuationUnderThreshold) {
  auto model = GenerativeModel::make(std::make_shared<MockBackend>(),
                                     /* allowed_repeats= */ {},
                                     /* punctuation_tokens= */ {1},
                                     /* punctuation_repeat_threshold= */ 0.8);

  auto output = model->generate(/* input_tokens= */ {0}, /* prompt= */ {},
                                /* n_predictions= */ 3,
                                /* beam_width= */ 1);

  std::vector<uint32_t> expected_output = {1, 2, 3};

  ASSERT_EQ(output, expected_output);
}

TEST(BeamSearchDecoding, AllowRepeatPunctuationOverThreshold) {
  auto model = GenerativeModel::make(std::make_shared<MockBackend>(),
                                     /* allowed_repeats= */ {},
                                     /* punctuation_tokens= */ {1},
                                     /* punctuation_repeat_threshold= */ 0.3);

  auto output = model->generate(/* input_tokens= */ {0}, /* prompt= */ {},
                                /* n_predictions= */ 3,
                                /* beam_width= */ 1);

  std::vector<uint32_t> expected_output = {1, 2, 1};

  ASSERT_EQ(output, expected_output);
}

TEST(BeamSearchDecoding, FindBestPath) {
  auto model = GenerativeModel::make(std::make_shared<MockBackend>(),
                                     /* allowed_repeats= */ {0, 1, 2, 3},
                                     /* punctuation_tokens= */ {},
                                     /* punctuation_repeat_threshold= */ 0.8);

  auto output = model->generate(/* input_tokens= */ {0}, /* prompt */ {},
                                /* n_predictions= */ 3,
                                /* beam_width= */ 2);

  // The greedy path would be 1 -> 2 -> 1, however the score for 1 -> 3 -> 2 is
  // better overall because 3 -> 2 is very good compared to 2 -> 1, whereas 1 ->
  // 3 is only slightly worse than 1 -> 2.
  std::vector<uint32_t> expected_output = {1, 3, 2};

  ASSERT_EQ(output, expected_output);
}

}  // namespace thirdai::bolt::tests