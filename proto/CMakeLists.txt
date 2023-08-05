find_package(Protobuf REQUIRED)

set(THIRDAI_PROTO_DEFS
    hashing/hashing.proto
    hashtable/hashtable.proto
    bolt/loss.proto
    bolt/model.proto
    bolt/neuron_index.proto
    bolt/ops.proto
    bolt/optimizers.proto
    bolt/parameter.proto)

protobuf_generate_cpp(PROTO_SRCS PROTO_HEADERS ${THIRDAI_PROTO_DEFS})
add_library(proto ${PROTO_HEADERS} ${PROTO_SRCS})
