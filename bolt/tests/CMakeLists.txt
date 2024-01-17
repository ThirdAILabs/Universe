enable_testing()

include(GoogleTest)

set(BOLT_NN_TESTS
    ComputationScheduleTests
    ComparativeLossTests
    FullyConnectedModelTests
    TensorTests
    InvalidModelTests
    TrainerTests
    ReduceLROnPlateauTests
    RootCauseAnalysisTests
    DlrmAttentionTests
    UpdateSparsityTests
    SaveLoadTests
    BeamSearchDecodingTests
    SwitchOpTests
    ArchiveSerializationTests
    EmbFcInferenceTests
    CosineLRScheduleTests)

foreach(BOLT_NN_TEST IN LISTS BOLT_NN_TESTS)
  add_executable(${BOLT_NN_TEST} ${BOLT_NN_TEST}.cc)
  target_link_libraries(${BOLT_NN_TEST} PRIVATE thirdai gtest_main)
  gtest_discover_tests(${BOLT_NN_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
