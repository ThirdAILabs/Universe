if(THIRDAI_BAKE_VERSION)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Version.h.in
                 ${CMAKE_CURRENT_BINARY_DIR}/Version.h @ONLY)
endif(THIRDAI_BAKE_VERSION)
