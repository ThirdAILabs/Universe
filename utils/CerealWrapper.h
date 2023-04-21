#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>

/**
 * For windows we CEREAL_REGISTER_TYPE must occur in the header files. However
 * registering in header files significantly slows down compilation. Because
 * Linux/Mac do not have the issue requiring it be done in a header file these
 * wrappers are a hack so that if CEREAL_REGISTER_TYPE_HEADER is used in the
 * header file and CEREAL_REGISTER_TYPE_SOURCE is used in the cc file then
 * depending on the os only one will actually resolve to CEREAL_REGISTER_TYPE at
 * compile time.
 *
 * https://uscilab.github.io/cereal/polymorphism.html
 */

#if _WIN32
#define CEREAL_REGISTER_TYPE_HEADER(type) CEREAL_REGISTER_TYPE(type)
#else
#define CEREAL_REGISTER_TYPE_HEADER(type)
#endif

#if _WIN32
#define CEREAL_REGISTER_TYPE_SOURCE(type)
#else
#define CEREAL_REGISTER_TYPE_SOURCE(type) CEREAL_REGISTER_TYPE(type)
#endif