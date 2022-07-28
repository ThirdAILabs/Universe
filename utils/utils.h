#pragma once
#include <cstdlib>
#include <iostream>

#define BOLT_TRACE(variable)                                   \
  do {                                                         \
    if (std::getenv("BOLT_TRACE")) {                           \
      std::cerr << #variable << ": " << variable << std::endl; \
    }                                                          \
  } while (0)
