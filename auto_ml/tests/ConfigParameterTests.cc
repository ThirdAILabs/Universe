#include "gtest/gtest.h"
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/config/Parameter.h>

namespace thirdai::automl::config::tests {

TEST(ConfigParameterTests, ConstantBoolean) {
  json config = R"(
    {"key": true}
  )"_json;

  ASSERT_EQ(booleanParameter(config, "key", {}), true);
}

TEST(ConfigParameterTests, UserSpecifiedBoolean) {
  json config = R"(
    {"key": {"param_name": "user_key"}}
  )"_json;

  ArgumentMap inputs;
  inputs.insert<bool>("user_key", false);

  ASSERT_EQ(booleanParameter(config, "key", inputs), false);
}

TEST(ConfigParameterTests, OptionMappedBoolean) {
  json config = R"(
    {
      "key": {
        "param_name": "user_key",
        "param_options": {
          "good": true,
          "bad": false
        }
      }
    }
  )"_json;

  {
    ArgumentMap inputs;
    inputs.insert<std::string>("user_key", "good");
    ASSERT_EQ(booleanParameter(config, "key", inputs), true);
  }

  {
    ArgumentMap inputs;
    inputs.insert<std::string>("user_key", "bad");
    ASSERT_EQ(booleanParameter(config, "key", inputs), false);
  }
}

TEST(ConfigParameterTests, ConstantInteger) {
  json config = R"(
    {"key": 249824}
  )"_json;

  ASSERT_EQ(integerParameter(config, "key", {}), 249824);
}

TEST(ConfigParameterTests, UserSpecifiedInteger) {
  json config = R"(
    {"key": {"param_name": "user_key"}}
  )"_json;

  ArgumentMap inputs;
  inputs.insert<uint32_t>("user_key", 64092);

  ASSERT_EQ(integerParameter(config, "key", inputs), 64092);
}

TEST(ConfigParameterTests, OptionMappedInteger) {
  json config = R"(
    {
      "key": {
        "param_name": "user_key",
        "param_options": {
          "one": 1,
          "two": 2
        }
      }
    }
  )"_json;

  {
    ArgumentMap inputs;
    inputs.insert<std::string>("user_key", "one");
    ASSERT_EQ(integerParameter(config, "key", inputs), 1);
  }

  {
    ArgumentMap inputs;
    inputs.insert<std::string>("user_key", "two");
    ASSERT_EQ(integerParameter(config, "key", inputs), 2);
  }
}

TEST(ConfigParameterTests, ConstantFloat) {
  json config = R"(
    {"key": 9248.429}
  )"_json;

  ASSERT_FLOAT_EQ(floatParameter(config, "key", {}), 9248.429);
}

TEST(ConfigParameterTests, UserSpecifiedFloat) {
  json config = R"(
    {"key": {"param_name": "user_key"}}
  )"_json;

  ArgumentMap inputs;
  inputs.insert<float>("user_key", 72340.428);

  ASSERT_FLOAT_EQ(floatParameter(config, "key", inputs), 72340.428);
}

TEST(ConfigParameterTests, OptionMappedFloat) {
  json config = R"(
    {
      "key": {
        "param_name": "user_key",
        "param_options": {
          "half": 0.5,
          "third": 0.33
        }
      }
    }
  )"_json;

  {
    ArgumentMap inputs;
    inputs.insert<std::string>("user_key", "half");
    ASSERT_FLOAT_EQ(floatParameter(config, "key", inputs), 0.5);
  }

  {
    ArgumentMap inputs;
    inputs.insert<std::string>("user_key", "third");
    ASSERT_FLOAT_EQ(floatParameter(config, "key", inputs), 0.33);
  }
}

TEST(ConfigParameterTests, ConstantString) {
  json config = R"(
    {"key": "something"}
  )"_json;

  ASSERT_EQ(stringParameter(config, "key", {}), "something");
}

TEST(ConfigParameterTests, UserSpecifiedString) {
  json config = R"(
    {"key": {"param_name": "user_key"}}
  )"_json;

  ArgumentMap inputs;
  inputs.insert<std::string>("user_key", "hello");

  ASSERT_EQ(stringParameter(config, "key", inputs), "hello");
}

TEST(ConfigParameterTests, OptionMappedString) {
  json config = R"(
    {
      "key": {
        "param_name": "user_key",
        "param_options": {
          "long": "12345678",
          "short": "123"
        }
      }
    }
  )"_json;

  {
    ArgumentMap inputs;
    inputs.insert<std::string>("user_key", "long");
    ASSERT_EQ(stringParameter(config, "key", inputs), "12345678");
  }

  {
    ArgumentMap inputs;
    inputs.insert<std::string>("user_key", "short");
    ASSERT_EQ(stringParameter(config, "key", inputs), "123");
  }
}

}  // namespace thirdai::automl::config::tests