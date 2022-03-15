#pragma once

#include <memory>

namespace sv {

/// @brief A simple type-erased base class
/// @example
/// class Interface : public Poly<Base>  {
///    using Poly::Poly;
///    // actual interface here
/// }
template <typename Base>
class Poly {
 public:
  Poly() noexcept = default;

  template <typename T>
  Poly(T x) : self_(std::make_shared<T>(std::move(x))) {}

  explicit operator bool() const { return self_ != nullptr; }

 protected:
  std::shared_ptr<const Base> self_{nullptr};
};

}  // namespace sv
