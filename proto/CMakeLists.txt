set(THIRDAI_PROTO_DEFS
    hashing/hashing.proto
    hashtable/hashtable.proto
    bolt/loss.proto
    bolt/model.proto
    bolt/neuron_index.proto
    bolt/ops.proto
    bolt/optimizers.proto
    bolt/parameter.proto
    data/transformations/binning.proto
    data/transformations/categorical_temporal.proto
    data/transformations/cold_start.proto
    data/transformations/date.proto
    data/transformations/feature_hash.proto
    data/transformations/mach_label.proto
    data/transformations/string_cast.proto
    data/transformations/string_concat.proto
    data/transformations/string_hash.proto
    data/transformations/string_id_lookup.proto
    data/transformations/text_tokenizer.proto
    data/encoders.proto
    data/tokenizers.proto
    data/transformations.proto)

protobuf_generate_cpp(PROTO_SRCS PROTO_HEADERS ${THIRDAI_PROTO_DEFS})
add_library(proto ${PROTO_HEADERS} ${PROTO_SRCS})
