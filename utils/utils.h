#pragma once
#include <iostream>

#define BOLT_TRACE(variable)                                 \
  do {                                                       \
    std::cerr << #variable << ": " << variable << std::endl; \
  } while (0)
