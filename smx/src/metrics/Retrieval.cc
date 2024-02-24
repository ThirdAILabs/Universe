#include "Metrics.h"
#include <smx/src/tensor/CsrTensor.h>
#include <algorithm>
#include <stdexcept>

namespace thirdai::smx {

using IdAndScore = std::pair<uint32_t, float>;

struct HighestScore {
  bool operator()(const IdAndScore& a, const IdAndScore& b) const {
    return a.second > b.second;
  }
};

std::vector<IdAndScore> topk(const uint32_t* ids, const float* scores,
                             size_t len, size_t k) {
  std::vector<IdAndScore> queue;
  queue.reserve(k + 1);

  HighestScore cmp;

  for (size_t i = 0; i < len; i++) {
    uint32_t id = ids[i];
    float score = scores[i];

    if (queue.size() < k || queue.front().second < score) {
      queue.emplace_back(id, score);
      std::push_heap(queue.begin(), queue.end(), cmp);
    }

    if (queue.size() > k) {
      std::pop_heap(queue.begin(), queue.end(), cmp);
      queue.pop_back();
    }
  }

  return queue;
}

uint32_t truePositives(const uint32_t* scores_i, const float* scores_v,
                       size_t scores_len, const uint32_t* labels_i,
                       const float* labels_v, size_t labels_len, size_t k) {
  auto topk_scores = topk(scores_i, scores_v, scores_len, k);

  uint32_t true_positives = 0;

  const uint32_t* labels_i_end = labels_i + labels_len;
  for (auto [id, _] : topk_scores) {
    const auto* loc = std::find(labels_i, labels_i_end, id);
    if (loc != labels_i_end && labels_v[loc - labels_i] > 0) {
      true_positives++;
    }
  }

  return true_positives;
}

float precision(const CsrTensorPtr& scores, const CsrTensorPtr& labels,
                size_t k) {
  const uint32_t* scores_o = scores->rowOffsets()->data<uint32_t>();
  const uint32_t* scores_i = scores->colIndices()->data<uint32_t>();
  const float* scores_v = scores->colValues()->data<float>();

  const uint32_t* labels_o = labels->rowOffsets()->data<uint32_t>();
  const uint32_t* labels_i = labels->colIndices()->data<uint32_t>();
  const float* labels_v = labels->colValues()->data<float>();

  size_t rows = scores->nRows();

  uint32_t true_positives = 0, total_preds = 0;

  for (size_t n = 0; n < rows; n++) {
    uint32_t scores_start = scores_o[n], scores_end = scores_o[n + 1];
    uint32_t labels_start = labels_o[n], labels_end = labels_o[n + 1];

    true_positives += truePositives(
        /*scores_i=*/scores_i + scores_start,
        /*scores_v=*/scores_v + scores_start,
        /*scores_len=*/scores_end - scores_start,
        /*labels_i=*/labels_i + labels_start,
        /*labels_v=*/labels_v + labels_start,
        /*labels_len=*/labels_end - labels_start, k);

    total_preds += std::min<uint32_t>(scores_end - scores_start, k);
  }

  return static_cast<float>(true_positives) / total_preds;
}

float precision(const TensorPtr& scores, const CsrTensorPtr& labels, size_t k) {
  CHECK(scores->shape() == labels->shape(),
        "Scores and labels must have the same shape.");
  if (scores->isSparse()) {
    return precision(csr(scores), labels, k);
  }
  throw std::runtime_error(
      "precision@k is only implemented for sparse scores.");
}

float recall(const CsrTensorPtr& scores, const CsrTensorPtr& labels, size_t k) {
  const uint32_t* scores_o = scores->rowOffsets()->data<uint32_t>();
  const uint32_t* scores_i = scores->colIndices()->data<uint32_t>();
  const float* scores_v = scores->colValues()->data<float>();

  const uint32_t* labels_o = labels->rowOffsets()->data<uint32_t>();
  const uint32_t* labels_i = labels->colIndices()->data<uint32_t>();
  const float* labels_v = labels->colValues()->data<float>();

  size_t rows = scores->nRows();

  uint32_t true_positives = 0, total_labels = 0;

  for (size_t n = 0; n < rows; n++) {
    uint32_t scores_start = scores_o[n], scores_end = scores_o[n + 1];
    uint32_t labels_start = labels_o[n], labels_end = labels_o[n + 1];

    true_positives += truePositives(
        /*scores_i=*/scores_i + scores_start,
        /*scores_v=*/scores_v + scores_start,
        /*scores_len=*/scores_end - scores_start,
        /*labels_i=*/labels_i + labels_start,
        /*labels_v=*/labels_v + labels_start,
        /*labels_len=*/labels_end - labels_start, k);

    uint32_t positive_labels = 0;
    for (size_t i = labels_start; i < labels_end; i++) {
      if (labels_v[i] > 0) {
        positive_labels++;
      }
    }
  }

  return static_cast<float>(true_positives) / total_labels;
}

float recall(const TensorPtr& scores, const CsrTensorPtr& labels, size_t k) {
  CHECK(scores->shape() == labels->shape(),
        "Scores and labels must have the same shape.");
  if (scores->isSparse()) {
    return recall(csr(scores), labels, k);
  }
  throw std::runtime_error("recall@k is only implemented for sparse scores.");
}

}  // namespace thirdai::smx