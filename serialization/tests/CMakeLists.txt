enable_testing()

include(GoogleTest)

set(ARCHIVE_TESTS
    ArchiveMapTests
    ArchiveListTests
    ArchiveValueTests
    ParameterReferenceTests)

foreach(ARCHIVE_TEST IN LISTS ARCHIVE_TESTS)
  add_executable(${ARCHIVE_TEST} ${ARCHIVE_TEST}.cc)
  target_link_libraries(${ARCHIVE_TEST} PRIVATE thirdai gtest_main)
  gtest_discover_tests(${ARCHIVE_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
