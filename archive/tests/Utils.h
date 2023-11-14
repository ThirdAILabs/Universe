#pragma once
#include "gtest/gtest.h"

namespace thirdai::ar::tests {

// NOLINTNEXTLINE (clang-tidy doesn't like macros)
#define CHECK_EXCEPTION(statement, msg, exception_type)             \
  try {                                                             \
    statement;                                                      \
    FAIL() << "Expected exception to be thrown.";                   \
  } catch (const exception_type& err) {                             \
    ASSERT_EQ(err.what(), std::string(msg));                        \
  } catch (...) {                                                   \
    FAIL() << "Expected different type of exception to be thrown."; \
  }

}  // namespace thirdai::ar::tests