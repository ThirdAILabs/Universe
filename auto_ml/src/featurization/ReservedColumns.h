#pragma once

#include <auto_ml/src/featurization/DataTypes.h>
#include <regex>
#include <stdexcept>

namespace thirdai::automl {

bool isReservedColumnName(const std::string& name);

void checkNoReservedColumnNames(const ColumnDataTypes& data_types);

std::string textOutputColumn(const std::string& input_column_name);

std::string categoricalOutputColumn(const std::string& input_column_name);

std::string binningOutputColumn(const std::string& input_column_name);

std::string sequenceOutputColumn(const std::string& input_column_name);

std::string dateOutputColumn(const std::string& input_column_name);

std::string temporalItemIdsOutput(const std::string& input_column_name);

std::string temporalTrackingOutput(uint32_t temporal_id);

const std::string TABULAR_COLUMNS_OUTPUT = "__tabular_columns__";

const std::string TIMESTAMP_OUTPUT = "__timestamp__";

const std::string FEATURIZED_INDICES = "__featurized_input_indices__";

const std::string FEATURIZED_VALUES = "__featurized_input_values__";

const std::string FEATURIZED_LABELS = "__labels__";

const std::string MACH_BUCKETS = "__mach__labels__";

const std::string MACH_DOC_IDS = "__mach_doc_ids__";

const std::string GRAPH_NBR_FEATURES = "__graph_nbr_features__";

const std::string GRAPH_NBR_IDS = "__graph_nbr_ids__";

const std::string RECURRENT_SEQUENCE = "__sequential_target__";

}  // namespace thirdai::automl