
#include <memory>
#include <string>

namespace thirdai::bolt {

class TemporalTracking {
 public:
  virtual ~TemporalTracking() = default;
};

using TemporalTrackingPtr = std::shared_ptr<TemporalTracking>;

struct CategoricalTemporalTracking : public TemporalTracking {
 public:
  CategoricalTemporalTracking(std::string column_name, uint32_t history_length,
                              bool use_current_row)
      : tracked_column_name(std::move(column_name)),
        history_length(history_length),
        use_current_row(use_current_row) {}

  std::string tracked_column_name;
  uint32_t history_length;
  bool use_current_row;
};

using CategoricalTemporalTrackingPtr =
    std::shared_ptr<CategoricalTemporalTracking>;

class NumericalTemporalTracking : public TemporalTracking {
 public:
  NumericalTemporalTracking(std::string column_name, uint32_t history_length,
                            bool use_current_row)
      : tracked_column_name(std::move(column_name)),
        history_length(history_length),
        use_current_row(use_current_row) {}

  std::string tracked_column_name;
  uint32_t history_length;
  bool use_current_row;
};

using NumericalTemporalTrackingPtr = std::shared_ptr<NumericalTemporalTracking>;

}  // namespace thirdai::bolt