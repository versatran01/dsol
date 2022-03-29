#pragma once

#include <opencv2/core/types.hpp>

#include "sv/dsol/dim.h"
#include "sv/util/eigen.h"

namespace sv::dsol {

/// @brief Image gradient block of patch Hessian
struct PatchHessian {
  using Matrix2d = Eigen::Matrix2d;
  using Vector2d = Eigen::Vector2d;

  using ArrayKd = ArrayMNd<Dim::kPatch, 1>;
  using Matrix2Kd = MatrixMNd<2, Dim::kPatch>;
  using MatrixK2d = MatrixMNd<Dim::kPatch, 2>;

  Matrix2d ItI{Matrix2d::Zero()};  // It * I
  Vector2d Itr{Vector2d::Zero()};  // It * r
  double r2{};                     // r * r
  double wr2{};                    // w * r * r

  /// @brief Add only the intensity part
  void AddI(const Vector2d& It, double r, double w) noexcept;
  void SetI(const Matrix2Kd& It, const ArrayKd& r, const ArrayKd& w) noexcept;
};

/// @brief Single frame Hessian with affine
/// Block structure is
/// [It*I, It*A] | [It*r]
/// [At*I, At*A] | [At*r]
struct PatchHessian1 final : public PatchHessian {
  Matrix2d ItA{Matrix2d::Zero()};
  Matrix2d AtA{Matrix2d::Zero()};
  Vector2d Atr{Vector2d::Zero()};

  /// @brief Accumulate patch Hessian
  void AddA(const Vector2d& It,
            const Vector2d& At,
            double r,
            double w) noexcept;
  /// @brief Batched version of Add(), 15~20% faster
  void SetA(const Matrix2Kd& It,
            const Matrix2Kd& At,
            const ArrayKd& r,
            const ArrayKd& w) noexcept;
};

/// @brief Two frames patch hessian
/// Block Structure is
/// [[It*I,  It*A0], [ It*I,  It*A1]] | [ It*r]
/// [[    , A0t*A0], [A0t*I, A0t*A1]] | [A0t*r]
/// [[            ], [ It*I,  It*A1]] | [ It*r]
/// [[            ], [A1t*I, A1t*A1]] | [A1t*r]
struct PatchHessian2 final : public PatchHessian {
  Matrix2d ItA0{Matrix2d::Zero()};
  Matrix2d ItA1{Matrix2d::Zero()};

  Matrix2d A0tA0{Matrix2d::Zero()};
  Matrix2d A0tA1{Matrix2d::Zero()};
  Matrix2d A1tA1{Matrix2d::Zero()};

  Vector2d A0tr{Vector2d::Zero()};
  Vector2d A1tr{Vector2d::Zero()};

  void AddA(const Vector2d& It,
            const Vector2d& A0t,
            const Vector2d& A1t,
            double r,
            double w) noexcept;

  void SetA(const Matrix2Kd& It,
            const Matrix2Kd& A0t,
            const Matrix2Kd& A1t,
            const ArrayKd& r,
            const ArrayKd& w) noexcept;
};

/// @brief Base frame Hessian, contains types, dimension info, num and costs
struct FrameHessian {
  using Vector10d = MatrixMNd<Dim::kFrame, 1>;
  using Matrix10d = MatrixMNd<Dim::kFrame, Dim::kFrame>;
  using Matrix26d = MatrixMNd<2, Dim::kPose>;

  int n{};     // num costs
  double c{};  // cost (r^2)

  int num_costs() const noexcept { return n; }
  double cost() const noexcept { return c; }
  bool Ok() const noexcept { return n > 0; }
  double MeanCost() const noexcept { return c / n; }

  void Reset() noexcept {
    n = 0;
    c = 0;
  }
};

/// @brief Single frame hessian (upper triangular), used in aligner
/// Block structure is
/// [Gt*(It*I)*G, Gt*(It*A)] | [Gt*(It*r)]
/// [   (At*I)*G,    (At*A)] | [   (At*r)]
/// We keep the lower-triangular part, according to
/// https://eigen.tuxfamily.org/dox/classEigen_1_1LLT.html
struct FrameHessian1 final : public FrameHessian {
  Matrix10d H{Matrix10d::Zero()};  // lhs:  Jt*W*J
  Vector10d b{Vector10d::Zero()};  // rhs: -Jt*W*r

  /// @brief Operator +/+=
  FrameHessian1& operator+=(const FrameHessian1& rhs) noexcept;
  friend FrameHessian1 operator+(FrameHessian1 lhs,
                                 const FrameHessian1& rhs) noexcept {
    return lhs += rhs;
  }

  /// @brief Add inner frame hessian
  /// @param affine_offset should be 0 or 2. If < 0 means not using affine
  void AddPatchHess(const PatchHessian1& ph,
                    const Matrix26d& G,
                    int affine_offset) noexcept;
};

/// @brief Two frames hessian, used in adjuster
/// Block Structure is
/// [[G0t*(It*I)*G0, G0t*(It*A0)], [G0t*(It*I)*G1, G0t*(It*A1)]] | [G0t*(It*r)]
/// [[   (A0t*I)*G0,    (A0t*A0)], [   (A0t*I)*G1,    (A0t*A1)]] | [   (A0t*r)]
/// [[G1t*(It*I)*G0, G1t*(It*A0)], [G1t*(It*I)*G1, G1t*(It*A1)]] | [G1t*(It*r)]
/// [[   (A1t*I)*G0,    (A1t*A0)], [   (A1t*I)*G1,    (A1t*A1)]] | [   (A1t*r)]
/// We keep the upper-triangular part because it's easy to reason about even
/// though it is slower
struct FrameHessian2 final : public FrameHessian {
  // frame indices (multiply by dim to get matrix index)
  int i_{-1};
  int j_{-1};

  Matrix10d Hii{Matrix10d::Zero()};
  Matrix10d Hij{Matrix10d::Zero()};
  Matrix10d Hjj{Matrix10d::Zero()};
  Vector10d bi{Vector10d::Zero()};
  Vector10d bj{Vector10d::Zero()};

  FrameHessian2() = default;
  FrameHessian2(int fi, int fj);
  explicit FrameHessian2(const Eigen::Vector2i& fij)
      : FrameHessian2(fij[0], fij[1]) {}

  /// @brief Actual index into full hessian matrix
  int ii() const noexcept { return i_ * Dim::kFrame; }
  int jj() const noexcept { return j_ * Dim::kFrame; }

  /// @brief Repr
  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const FrameHessian2& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Operator +/+=
  FrameHessian2& operator+=(const FrameHessian2& rhs);
  friend FrameHessian2 operator+(FrameHessian2 lhs, const FrameHessian2& rhs) {
    return lhs += rhs;
  }

  void AddPatchHess(const PatchHessian2& ph,
                    const Matrix26d& G0,
                    const Matrix26d& G1,
                    int affine_offset) noexcept;
};

/// @brief Dynamic-sized frame hessian
/// @note This class is not meant to be copied or moved
struct FrameHessianX : public FrameHessian {
 private:
  std::vector<double> data_;  // actual storage of data

 public:
  MatrixXdMap Hpp{nullptr, 0, 0};
  VectorXdMap bp{nullptr, 0};

  FrameHessianX() = default;
  explicit FrameHessianX(int nframes) { MapFrames(nframes); }

  bool empty() const noexcept { return data_.empty(); }
  size_t size() const noexcept { return data_.size(); }
  size_t capacity() const noexcept { return data_.capacity(); }

  const auto& data() const noexcept { return data_; }
  double* data_ptr() noexcept { return data_.data(); }
  auto storage() noexcept { return VectorXdMap(data_.data(), size()); }

  int dim_frames() const noexcept { return static_cast<int>(bp.size()); }
  int num_frames() const noexcept { return dim_frames() / Dim::kFrame; }

  void ReserveData(int size) noexcept { data_.reserve(size); }
  void ResizeData(int size) noexcept { data_.resize(size); }
  void ResetData() noexcept;

  /// @brief Update internal Eigen Maps
  int MapFrames(int nframes);

  void Scale(double s = 1.0) noexcept;
  double DiagSum() const noexcept { return Hpp.diagonal().sum(); }

  /// @brief Add value along diagnoal of Hpp
  void AddDiag(int ind, int size, double val);

  /// @brief Fix gauge by adding a large value (lambda) to the diagonal of the
  /// hessian block of the first frame.
  void FixGauge(double lambda, bool fix_scale);

  /// @brief A self-adjoint view of Hpp
  auto HppAdj() noexcept { return Hpp.selfadjointView<Eigen::Lower>(); }
};

/// @brief Marginalized prior on first n-1 frames
struct PriorFrameHessian final : public FrameHessianX {
  using FrameHessianX::FrameHessianX;
};

/// @brief Stores results after schur complement
struct SchurFrameHessian final : public FrameHessianX {
  using FrameHessianX::FrameHessianX;

  /// @brief Add prior, with scale
  void AddPriorHess(const PriorFrameHessian& prior);

  /// @brief Solve for frame parameters
  void Solve(VectorXdRef xp, VectorXdRef yp);

  /// @brief Marginalize frame
  void MargFrame(PriorFrameHessian& prior, int fid);
};

/// @brief Full Block Hessian with extra storage used for solving
struct FramePointHessian final : public FrameHessianX {
  static constexpr int kPointDim = Dim::kPoint;

  // Data
  bool ready_{false};  // whether block is ready to be solved

  // These won't change durning solving
  // p - pose, m - map
  // They should really be named f - frame, d - (inverse) depth
  // [ Hpp, Hpm ] * [ xp ] = [ bp ]
  // [ Hmp, Hmm ]   [ xm ] = [ bm ]

  // From base class
  // MatrixXdMap Hpp_;
  // VectorXdMap bp_;

  MatrixXdMap Hpm{nullptr, 0, 0};   // full
  VectorXdMap Hmm{nullptr, 0};      // diagonal, stores inverse
  VectorXdMap Hmm_inv{nullptr, 0};  // map to the same storage as Hmm
  VectorXdMap bm{nullptr, 0};

  // Solution
  VectorXdMap xp{nullptr, 0};
  VectorXdMap xm{nullptr, 0};

  FramePointHessian() = default;
  FramePointHessian(int nframes, int npoints) { MapFull(nframes, npoints); }

  /// @brief Repr
  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os,
                                  const FramePointHessian& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Compute total storage size
  static int CalcDataSize(int frame_dim, int point_dim) noexcept;

  /// @brief info
  int dim_full() const noexcept { return dim_frames() + dim_points(); }
  int dim_points() const noexcept { return static_cast<int>(bm.size()); }
  int num_points() const noexcept { return dim_points() / kPointDim; }

  /// @brief Block of frame i in xp
  Eigen::Map<const Vector10d> XpAt(int i) const noexcept;
  Eigen::Map<Vector10d> XpAt(int i) noexcept;

  /// @brief Reset storage and state
  void ResetFull(double Hpp_diag = 0.0) noexcept;
  /// @brief Reserve storage
  void ReserveFull(int nframes, int npoints);
  /// @brief Update internal Eigen Maps
  int MapFull(int nframes, int npoints);

  /// @brief Add frame Hessian to full Hessian
  void AddFrameHess(const FrameHessian2& fh);
  /// @brief Add patch Hessian
  void AddPatchHess(const PatchHessian2& ph,
                    const Matrix26d& G0,
                    const Matrix26d& G1,
                    const Eigen::Vector2d& Gd,
                    const Eigen::Vector3i& ijh,
                    int affine_offset) noexcept;

  /// @brief Invert Hmm and make Hpp symmetric
  void Prepare();
  /// @brief Solve for frames and points
  void Solve();

  /// @brief Marginalize points (all or single frame)
  int MargPointsAll(SchurFrameHessian& schur, int gsize = 0) const;
  int MargPointsRange(SchurFrameHessian& schur,
                      const cv::Range& range,
                      int gsize = 0);
};

/// @brief Marginalize points onto frames, return number of points marginalized
/// @param Hpp could be lower triangular
/// @param Hsc is symmetric
int MargPointsToFrames(const MatrixXdCRef& Hpp,
                       const MatrixXdCRef& Hpm,
                       const VectorXdCRef& Hmm_inv,
                       const VectorXdCRef& bp,
                       const VectorXdCRef& bm,
                       MatrixXdRef Hsc,
                       VectorXdRef bsc,
                       int gsize = 0);

/// @brief PointHessian used in refiner
struct PointHessian {
 private:
  std::vector<double> data_;

 public:
  VectorXdMap H{nullptr, 0};  // diagonal
  VectorXdMap b{nullptr, 0};
  VectorXdMap x{nullptr, 0};

  PointHessian() = default;
  explicit PointHessian(int npoints) { MapPoints(npoints); }

  int dim_points() const noexcept { return static_cast<int>(b.size()); }
  int num_points() const noexcept { return dim_points() / Dim::kPoint; }

  /// @brief Update internal Eigen Maps
  int MapPoints(int npoints);
};

}  // namespace sv::dsol
