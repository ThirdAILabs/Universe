#pragma once

#include <iostream>
#include <optional>

constexpr char OPEN_CHAR = '[';
constexpr char CLOSE_CHAR = ']';
constexpr char DONE_CHAR = '=';
constexpr char TODO_CHAR = ' ';

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
    std::cout << '\r' << _description << ": " << std::flush;
    std::cout << OPEN_CHAR;
    for (uint32_t i = 0; i < BAR_SIZE; i++) {
      std::cout << TODO_CHAR;
    }
    std::cout << CLOSE_CHAR << " " << _prev_percent << "%";

    /**
     * The description for train is 'train epoch N' whereas the description for
     * evaluate is 'evaluate'. This means that if the train progress bar is
     * interrupted by the evaluation progress bar during validation, it won't
     * completely overwrite the previous training progress bar. This ensures
     * that it will clear the training progress bar if N < 1,000,000.
     */
    for (uint32_t i = 0; i < 10; i++) {
      std::cout << ' ';
    }
    std::cout << std::flush;
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

    std::cout << _description << ": " << OPEN_CHAR;

    // Fill ticks
    uint32_t new_ticks = (new_percent + 1) / 2;
    for (uint32_t i = 0; i < new_ticks; i++) {
      std::cout << DONE_CHAR;
    }

    // Fill the rest of the space
    for (uint32_t i = new_ticks; i < BAR_SIZE; i++) {
      std::cout << TODO_CHAR;
    }

    std::cout << CLOSE_CHAR << " " << new_percent << "%" << std::flush;

    _prev_ticks = new_ticks;
    _prev_percent = new_percent;
  }

  void close(const std::string& comment) {
    // This overwrites the progress bar with its closing comment so that the bar
    // is not displayed after it completes.
    std::cout << "\r" << comment;

    /**
     * Clear out any additional information that's longer than the comment.
     * The +9 comes from:
     *    +2 => ': ' after the description (2)
     *    +2 => OPEN_CHAR + CLOSE_CHAR
     *    +5 => ' 100%' at the end of the bar when its complete.
     *
     * Adding the 9 gives us the length of everything already printed as part of
     * the bar, so we know if there is additional information that needs to be
     * overwritten.
     */
    uint32_t current_line_len = _description.size() + BAR_SIZE + 9;
    if (current_line_len > comment.size()) {
      for (uint32_t i = 0; i < current_line_len - comment.size(); i++) {
        std::cout << ' ';
      }
    }
    std::cout << '\n' << std::endl;
  }
};

using ProgressBarPtr = std::shared_ptr<ProgressBar>;
