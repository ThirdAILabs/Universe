#include <gtest/gtest.h>
#include <serialization/src/Archive.h>
#include <serialization/src/ParameterReference.h>
#include <serialization/tests/Utils.h>
#include <sstream>
#include <stdexcept>

namespace thirdai::ar::tests {

TEST(ParameterReferenceTests, CannotAccessParameterBeforeLoad) {
  std::vector<float> data = {1.0, 2.0, 3.0};

  ParameterReference ref(data, /* op= */ nullptr);

  CHECK_EXCEPTION(ref.loadedParameter(),
                  "Cannot access the parameter in a ParameterReference before "
                  "saving and loading.",
                  std::runtime_error)
}

TEST(ParameterReferenceTests, Serialization) {
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  auto ref = ParameterReference::make(data, /* op= */ nullptr);

  std::stringstream buffer;
  serialize(ref, buffer);
  auto loaded = deserialize(buffer);

  ASSERT_EQ(*loaded->param().loadedParameter(), data);

  const auto* loaded_data_ptr = loaded->param().loadedParameter()->data();

  std::vector<float> loaded_data = loaded->param().takeLoadedParameter();
  ASSERT_EQ(loaded_data, data);
  ASSERT_EQ(loaded_data.data(), loaded_data_ptr);  // Check take doesn't copy.
}

}  // namespace thirdai::ar::tests