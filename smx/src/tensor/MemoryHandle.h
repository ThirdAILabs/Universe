#pragma once

#include <memory>

namespace thirdai::smx {

class MemoryHandle {
 public:
  virtual void* ptr() const = 0;

  virtual size_t nbytes() const = 0;

  virtual ~MemoryHandle() = default;
};

using MemoryHandlePtr = std::shared_ptr<MemoryHandle>;

class DefaultMemoryHandle final : public MemoryHandle {
 public:
  explicit DefaultMemoryHandle(size_t nbytes)
      : _data(new uint8_t[nbytes]), _nbytes(nbytes) {}

  static auto allocate(size_t nbytes) {
    return std::make_shared<DefaultMemoryHandle>(nbytes);
  }

  void* ptr() const final { return _data.get(); }

  size_t nbytes() const final { return _nbytes; }

 private:
  std::unique_ptr<uint8_t[]> _data;
  size_t _nbytes;
};

}  // namespace thirdai::smx