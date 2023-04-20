#pragma once

#include <cereal/types/polymorphic.hpp>

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