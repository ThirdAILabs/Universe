#include "Config.h"
#include <fstream>

namespace thirdai::bolt {

std::vector<std::string> Split(const std::string& line, char delimeter) {
  std::vector<std::string> output;
  std::string curr = line;
  size_t loc;
  while ((loc = curr.find(delimeter)) != std::string::npos) {
    output.push_back(curr.substr(0, loc));
    curr = curr.substr(loc + 1);
  }
  output.push_back(curr);
  return output;
}

void ConfigReader::ParseConfig(const std::string& filename) {
  std::ifstream file(filename);

  std::string line;

  while (std::getline(file, line)) {
    if (std::regex_search(line, comment_re) ||
        std::regex_search(line, empty_re)) {
      continue;
    }

    std::smatch key;
    if (!std::regex_search(line, key, key_re)) {
      throw std::logic_error("Invaid syntax: " + line +
                             " > Line in config file should contain key");
    }
    std::string key_str = key.str();

    size_t loc = line.find('=');
    if (loc == std::string::npos) {
      throw std::logic_error("Invaid syntax: " + line +
                             " > Line in config file should contain '='");
    }

    std::string remainder = line.substr(loc);

    std::vector<std::string> values = Split(remainder, ',');
    if (values.empty()) {
      throw std::logic_error("Invaid syntax: " + line +
                             " > Line in config file should have values");
    }

    if (std::regex_search(values[0], string_re)) {
      std::vector<std::string> var_values;
      for (const auto& val : values) {
        std::smatch str_val;
        if (!std::regex_search(val, str_val, string_re)) {
          throw std::logic_error(
              "Invaid syntax: " + line +
              " > All values in string variable should be strings");
        }

        var_values.push_back(
            str_val.str().substr(1, str_val.str().length() - 2));
      }

      config_vars[key_str] = std::make_shared<StrValue>(std::move(var_values));
    } else if (std::regex_search(values[0], decimal_re)) {
      std::vector<double> var_values;
      for (const auto& val : values) {
        std::smatch decimal_val;
        if (!std::regex_search(val, decimal_val, decimal_re)) {
          throw std::logic_error(
              "Invaid syntax: " + line +
              " > All values in decimal variable should be decimals");
        }
        var_values.push_back(atof(decimal_val.str().c_str()));
      }

      config_vars[key_str] =
          std::make_shared<DoubleValue>(std::move(var_values));
    } else if (std::regex_search(values[0], int_re)) {
      std::vector<uint64_t> var_values;
      for (const auto& val : values) {
        std::smatch int_val;
        if (!std::regex_search(val, int_val, int_re)) {
          throw std::logic_error(
              "Invaid syntax: " + line +
              " > All values in integer variable should be integers");
        }
        var_values.push_back(atoll(int_val.str().c_str()));
      }

      config_vars[key_str] = std::make_shared<IntValue>(std::move(var_values));
    } else {
      throw std::logic_error(
          "Invaid syntax: " + line +
          " > Values after = in config did not match integer, decimal, or "
          "string value expressions");
    }
  }

  file.close();
}

uint64_t ConfigReader::IntVal(const std::string& key, uint32_t index) const {
  if (!config_vars.count(key)) {
    throw std::logic_error("Invaid key in config file: '" + key + "'");
  }
  return config_vars.at(key)->IntVal(index);
}

float ConfigReader::FloatVal(const std::string& key, uint32_t index) const {
  if (!config_vars.count(key)) {
    throw std::logic_error("Invaid key in config file: '" + key + "'");
  }
  return static_cast<float>(config_vars.at(key)->DoubleVal(index));
}

double ConfigReader::DoubleVal(const std::string& key, uint32_t index) const {
  if (!config_vars.count(key)) {
    throw std::logic_error("Invaid key in config file: '" + key + "'");
  }
  return config_vars.at(key)->DoubleVal(index);
}

const std::string& ConfigReader::StrVal(const std::string& key,
                                        uint32_t index) const {
  if (!config_vars.count(key)) {
    throw std::logic_error("Invaid key in config file: '" + key + "'");
  }
  return config_vars.at(key)->StrVal(index);
}

bool ConfigReader::ValExists(const std::string& key) const {
  return config_vars.count(key);
}

void ConfigReader::PrintConfigVals() {
  std::cout << "\033[1;34m====== Config Vars ======\033[0m" << std::endl;
  for (const auto& var : config_vars) {
    std::cout << "\033[1;34m" << var.first << "\033[0m => " << var.second
              << std::endl;
  }
  std::cout << "\033[1;34m=========================\033[0m" << std::endl;

  std::cout << std::endl;
}

std::ostream& operator<<(std::ostream& out, const ConfigValue& val) {
  return val.Print(out);
}

std::ostream& operator<<(std::ostream& out, const std::shared_ptr<ConfigValue>& val) {
  return val->Print(out);
}

}  // namespace thirdai::bolt
