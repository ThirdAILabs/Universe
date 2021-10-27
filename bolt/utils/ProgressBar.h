// The MIT License (MIT)
//
// Copyright (c) 2019 Luigi Pertoldi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// ============================================================================
//  ___   ___   ___   __    ___   ____  __   __   ___    __    ___
// | |_) | |_) / / \ / /`_ | |_) | |_  ( (` ( (` | |_)  / /\  | |_)
// |_|   |_| \ \_\_/ \_\_/ |_| \ |_|__ _)_) _)_) |_|_) /_/--\ |_| \_
//
// Very simple progress bar for c++ loops with internal running variable
//
// Author: Luigi Pertoldi
// Created: 3 dic 2016
//
// Notes: The bar must be used when there's no other possible source of output
//        inside the for loop
//
// Note: This version is slighly modified from the original open source version
// to use chars instead of strings for better efficiency. It also makes some
// coding styles specific changes.

#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

class ProgressBar {
 public:
  // default destructor
  ~ProgressBar() = default;

  // delete everything else
  ProgressBar(ProgressBar const&) = delete;
  ProgressBar& operator=(ProgressBar const&) = delete;
  ProgressBar(ProgressBar&&) = delete;
  ProgressBar& operator=(ProgressBar&&) = delete;

  explicit inline ProgressBar(int n, bool showbar = true);

  // reset bar to use it again
  inline void reset();
  // chose your style
  inline void setDoneChar(char sym) { _done_char = sym; }
  inline void setTodoChar(char sym) { _todo_char = sym; }
  inline void setOpeningBracketChar(char sym) { _opening_bracket_char = sym; }
  inline void setClosingBracketChar(char sym) { _closing_bracket_char = sym; }
  // to show only the percentage
  inline void showBar(bool flag = true) { _do_show_bar = flag; }
  // main function
  inline void update();

 private:
  int _progress;
  int _n_cycles;
  int _last_perc;
  bool _do_show_bar;
  bool _update_is_called;

  char _done_char;
  char _todo_char;
  char _opening_bracket_char;
  char _closing_bracket_char;
};

inline ProgressBar::ProgressBar(int n, bool showbar)
    : _progress(0),
      _n_cycles(n),
      _last_perc(0),
      _do_show_bar(showbar),
      _update_is_called(false),
      _done_char('='),
      _todo_char(' '),
      _opening_bracket_char('['),
      _closing_bracket_char(']') {}

inline void ProgressBar::reset() {
  _progress = 0;
  _update_is_called = false;
  _last_perc = 0;
}

inline void ProgressBar::update() {
  if (!_update_is_called) {
    if (_do_show_bar) {
      std::cout << _opening_bracket_char;
      for (int _ = 0; _ < 50; _++) {
        std::cout << _todo_char;
      }
      std::cout << _closing_bracket_char << " 0%";
    } else {
      std::cout << "0%";
    }
  }
  _update_is_called = true;

  int perc = 0;

  // compute percentage, if did not change, do nothing and return
  perc = _progress * 100. / (_n_cycles - 1);
  if (perc < _last_perc) {
    return;
  }

  // update percentage each unit
  if (perc == _last_perc + 1) {
    // erase the correct  number of characters
    if (perc <= 10) {
      std::cout << "\b\b" << perc << '%';
    } else if ((perc > 10 && perc < 100) || perc == 100) {
      std::cout << "\b\b\b" << perc << '%';
    }
  }
  if (_do_show_bar) {
    // update bar every ten units
    if (perc % 2 == 0) {
      // erase closing bracket
      std::cout << '\b';
      // erase trailing percentage characters
      if (perc < 10) {
        std::cout << "\b\b\b";
      } else if (perc >= 10 && perc < 100) {
        std::cout << "\b\b\b\b";
      } else if (perc == 100) {
        std::cout << "\b\b\b\b\b";
      }
      // erase 'todo_char'
      for (int j = 0; j < 50 - (perc - 1) / 2; ++j) {
        std::cout << '\b';
      }

      // add one additional 'done_char'
      if (perc == 0) {
        std::cout << _todo_char;
      } else {
        std::cout << _done_char;
      }
      // refill with 'todo_char'
      for (int j = 0; j < 50 - (perc - 1) / 2 - 1; ++j) {
        std::cout << _todo_char;
      }

      // readd trailing percentage characters
      std::cout << _closing_bracket_char << ' ' << perc << '%';
    }
  }
  _last_perc = perc;
  ++_progress;
  std::cout << std::flush;
}