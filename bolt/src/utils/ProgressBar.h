#pragma once

#include <iostream>

constexpr char OPEN = '[';
constexpr char CLOSE = ']';
constexpr char DONE = '=';
constexpr char TODO = ' ';

class ProgressBar {
 private:
  static constexpr uint32_t BAR_SIZE = 50;
  uint32_t _prev_ticks, _prev_steps, _prev_percent;
  const uint32_t _max_steps;
  bool _verbose;

 public:
  ProgressBar(const ProgressBar&) = delete;
  ProgressBar(ProgressBar&&) = delete;
  ProgressBar() = delete;

  ProgressBar& operator=(const ProgressBar&) = delete;
  ProgressBar& operator=(ProgressBar&&) = delete;

  explicit ProgressBar(uint32_t max_steps, bool verbose = true)
      : _prev_ticks(0),
        _prev_steps(0),
        _prev_percent(0),
        _max_steps(max_steps),
        _verbose(verbose) {
    if (!_verbose) {
      return;
    }
    std::cout << OPEN;
    for (uint32_t i = 0; i < BAR_SIZE; i++) {
      std::cout << TODO;
    }
    std::cout << CLOSE << " " << _prev_percent << "%" << std::flush;
  }

  void increment() {
    if (!_verbose) {
      return;
    }
    uint32_t new_percent = (++_prev_steps) * 100.0 / _max_steps;
    if (new_percent == _prev_percent) {
      return;
    }

    // Clear percent
    if (_prev_percent < 10) {
      std::cout << "\b\b";
    } else if (_prev_percent < 100) {
      std::cout << "\b\b\b";
    } else if (_prev_percent == 100) {
      std::cout << "\b\b\b\b";
    }
    // Clear space and close bracket
    std::cout << "\b\b";

    for (uint32_t i = 0; i < BAR_SIZE - _prev_ticks; i++) {
      std::cout << '\b';
    }

    uint32_t new_ticks = (new_percent + 1) / 2;

    for (uint32_t i = 0; i < new_ticks - _prev_ticks; i++) {
      std::cout << DONE;
    }

    for (uint32_t i = 0; i < BAR_SIZE - new_ticks; i++) {
      std::cout << TODO;
    }
    std::cout << CLOSE << " " << new_percent << "%" << std::flush;

    _prev_ticks = new_ticks;
    _prev_percent = new_percent;
  }
};