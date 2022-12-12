#pragma once

#include <iostream>
#include <optional>

constexpr char OPEN = '[';
constexpr char CLOSE = ']';
constexpr char DONE = '=';
constexpr char TODO = ' ';

class ProgressBar {
 private:
  static constexpr uint32_t BAR_SIZE = 50;
  uint32_t _prev_ticks, _prev_steps, _prev_percent;
  const uint32_t _max_steps;
  std::string _description;

 public:
  ProgressBar(const ProgressBar&) = delete;
  ProgressBar(ProgressBar&&) = delete;
  ProgressBar() = delete;

  ProgressBar& operator=(const ProgressBar&) = delete;
  ProgressBar& operator=(ProgressBar&&) = delete;

  explicit ProgressBar(std::string description, uint32_t max_steps)
      : _prev_ticks(0),
        _prev_steps(0),
        _prev_percent(0),
        _max_steps(max_steps),
        _description(std::move(description)) {
    std::cout << _description << ": " << std::flush;
    std::cout << OPEN;
    for (uint32_t i = 0; i < BAR_SIZE; i++) {
      std::cout << TODO;
    }
    std::cout << CLOSE << " " << _prev_percent << "%" << std::flush;
  }

  static std::optional<ProgressBar> makeOptional(bool verbose,
                                                 const std::string& description,
                                                 uint32_t max_steps) {
    if (!verbose) {
      return std::nullopt;
    }
    return std::make_optional<ProgressBar>(description, max_steps);
  }

  void increment() {
    uint32_t new_percent = (++_prev_steps) * 100.0 / _max_steps;
    if (new_percent == _prev_percent) {
      return;
    }

    // Go back to start of line
    std::cout << '\r';

    std::cout << _description << ": " << OPEN;

    // Fill ticks
    uint32_t new_ticks = (new_percent + 1) / 2;
    for (uint32_t i = 0; i < new_ticks; i++) {
      std::cout << DONE;
    }

    // Fill the rest of the space
    for (uint32_t i = new_ticks; i < BAR_SIZE; i++) {
      std::cout << TODO;
    }

    std::cout << CLOSE << " " << new_percent << "%" << std::flush;

    _prev_ticks = new_ticks;
    _prev_percent = new_percent;
  }

  void close(const std::string& comment) {
    std::cout << "\r" << comment;
    for (uint32_t i = 0; i < _description.size() + BAR_SIZE + 9; i++) {
      std::cout << ' ';
    }
    std::cout << "\n" << std::endl;
  }
};
