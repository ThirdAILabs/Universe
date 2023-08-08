file(GLOB PROTO_DEFS ${PROJECT_SOURCE_DIR}/proto/**/*.proto)


set(PROTO_INCLUDE "")
set(PROTO_OUTPUT_DIR "${PROJECT_BINARY_DIR}/proto")
set(PROTO_DEFS_LIST "")
set(PROTO_HEADERS "")
set(PROTO_SOURCES "")

foreach(PROTO_DEF IN LISTS PROTO_DEFS)
    file(RELATIVE_PATH REL_PROTO_DEF ${PROJECT_SOURCE_DIR} ${PROTO_DEF})

    get_filename_component(PROTO_DEF_DIR ${PROTO_DEF} DIRECTORY)
    set(PROTO_INCLUDE "${PROTO_INCLUDE} -I${PROTO_DEF_DIR}")

    get_filename_component(PROTO_NAME ${REL_PROTO_DEF} NAME_WE)
    
    set(HEADER "${PROTO_OUTPUT_DIR}/${PROTO_NAME}.pb.h")
    set(SOURCE "${PROTO_OUTPUT_DIR}/${PROTO_NAME}.pb.cc")

    set(PROTO_DEFS_LIST "${PROTO_DEFS_LIST} ${PROTO_DEF}")

    list(APPEND PROTO_HEADERS ${HEADER})
    list(APPEND PROTO_SOURCES ${SOURCE})
endforeach()


# set(PROTOC_CMD "protoc -I${PROJECT_SOURCE_DIR}/proto --cpp_out ${PROJECT_BINARY_DIR}/proto ${PROTO_DEFS_LIST}")
# execute_process(
#     COMMAND protoc "-I${PROJECT_SOURCE_DIR}/proto" "--cpp_out ${PROJECT_BINARY_DIR}/proto" ${PROTO_DEFS_LIST}
# )
exec_program(
    protoc ARGS ${PROTO_INCLUDE} "--cpp_out ${PROTO_OUTPUT_DIR}" ${PROTO_DEFS_LIST}
)

set_source_files_properties(${PROTO_HEADERS} ${PROTO_SOURCES} PROPERTIES GENERATED TRUE)

# protobuf_generate_cpp(PROTO_SOURCES PROTO_HEADERS ${PROTO_DEFS})

add_library(proto ${PROTO_HEADERS} ${PROTO_SOURCES})

# Cross compiling causes issues if find_package(Protobuf) is used because it automatically 
# sets the libprotobuf to that of the host machine, which causes link errors. 
# If we stop cross compiling macos arm wheels then we can replace most of this with:
# protobuf_generate_cpp(PROTO_SOURCES PROTO_HEADERS ${PROTO_DEFS})
# add_library(proto ${PROTO_HEADERS} ${PROTO_SOURCES})