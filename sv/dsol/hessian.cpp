#include "sv/dsol/hessian.h"

#include <Eigen/Cholesky>  // LLT and LDLT
#include <Eigen/LU>        // inverse

#include "sv/util/logging.h"
#include "sv/util/tbb.h"

namespace sv::dsol {

using Vector2d = Eigen::Vector2d;
using Vector3i = Eigen::Vector3i;
using LltLowerInplace = Eigen::LLT<MatrixXdRef, Eigen::Lower>;
using LdltLowerInplace = Eigen::LDLT<MatrixXdRef, Eigen::Lower>;

namespace {

constexpr int dp = Dim::kPose;
constexpr int df = Dim::kFrame;
constexpr int dd = Dim::kPoint;
constexpr int da = Dim::kAffine;

}  // namespace

/// ============================================================================
void PatchHessian::AddI(const Vector2d& It, double r, double w) noexcept {
  const auto wr = w * r;
  ItI.noalias() += (It * w) * It.transpose();
  Itr.noalias() += It * wr;
  r2 += r * r;
  wr2 += r * wr;
}

void PatchHessian::SetI(const Matrix2Kd& It,
                        const ArrayKd& r,
                        const ArrayKd& w) noexcept {
  const ArrayKd wr = w * r;
  ItI.noalias() = It * w.matrix().asDiagonal() * It.transpose();
  Itr.noalias() = It * wr.matrix();
  r2 = (r * r).sum();
  wr2 = (wr * r).sum();
}

/// ============================================================================
void PatchHessian1::AddA(const Vector2d& It,
                         const Vector2d& At,
                         double r,
                         double w) noexcept {
  ItA.noalias() += (It * w) * At.transpose();
  AtA.noalias() += (At * w) * At.transpose();
  Atr.noalias() += At * (w * r);
}

void PatchHessian1::SetA(const Matrix2Kd& It,
                         const Matrix2Kd& At,
                         const ArrayKd& r,
                         const ArrayKd& w) noexcept {
  const MatrixK2d wA = w.matrix().asDiagonal() * At.transpose();
  ItA.noalias() = It * wA;
  AtA.noalias() = At * wA;
  Atr.noalias() = At * (w * r).matrix();
}

/// ============================================================================
void PatchHessian2::AddA(const Vector2d& It,
                         const Vector2d& A0t,
                         const Vector2d& A1t,
                         double r,
                         double w) noexcept {
  const Vector2d Itw = It * w;
  ItA0.noalias() += Itw * A0t.transpose();
  ItA1.noalias() += Itw * A1t.transpose();

  A0tA0.noalias() += (A0t * w) * A0t.transpose();
  A0tA1.noalias() += (A0t * w) * A1t.transpose();
  A1tA1.noalias() += (A1t * w) * A1t.transpose();

  const auto wr = w * r;
  A0tr.noalias() += A0t * wr;
  A1tr.noalias() += A1t * wr;
}

void PatchHessian2::SetA(const Matrix2Kd& It,
                         const Matrix2Kd& A0t,
                         const Matrix2Kd& A1t,
                         const ArrayKd& r,
                         const ArrayKd& w) noexcept {
  const MatrixK2d wA0 = w.matrix().asDiagonal() * A0t.transpose();
  const MatrixK2d wA1 = w.matrix().asDiagonal() * A1t.transpose();
  ItA0.noalias() = It * wA0;
  ItA1.noalias() = It * wA1;

  A0tA0.noalias() = A0t * wA0;
  A0tA1.noalias() = A0t * wA1;
  A1tA1.noalias() = A1t * wA1;

  const ArrayKd wr = w * r;
  A0tr.noalias() = A0t * wr.matrix();
  A1tr.noalias() = A1t * wr.matrix();
}

/// ============================================================================
FrameHessian1& FrameHessian1::operator+=(const FrameHessian1& rhs) noexcept {
  if (rhs.n > 0) {
    H += rhs.H;
    b += rhs.b;
    n += rhs.n;
    c += rhs.c;
  }
  return *this;
}

void FrameHessian1::AddPatchHess(const PatchHessian1& ph,
                                 const Matrix26d& G,
                                 int affine_offset) noexcept {
  // [Gt*(It*I)*G, Gt*(It*A)] | [Gt*(It*r)]
  // [   (At*I)*G,    (At*A)] | [   (At*r)]

  ++n;
  // FrameHessian1 is 10x10, where the variable is ordered as follows
  // [pose, aff0, aff1]
  // Add pose block, which is always top-left 6x6
  // Gt * ItI * G
  H.topLeftCorner<dp, dp>().noalias() += G.transpose() * ph.ItI * G;
  // Gt * It * r
  b.head<dp>().noalias() -= G.transpose() * ph.Itr;
  // cost
  c += ph.r2;

  if (affine_offset < 0) return;

  // Depends on which camera, the affine parameters would be different
  // Therefore we use affine_offset to indicate where should we put the
  // corresponding Hessian block. affine_index is pose_dim + affine_offset, left
  // is pose_dim, right is pose_dim + affine_dim
  const int ia = dp + affine_offset;  // 6(L) or 8(R)

  // https://eigen.tuxfamily.org/dox/classEigen_1_1LLT.html
  // Performance: for best performance, it is recommended to use a column-major
  // storage format with the Lower triangular part (the default), or,
  // equivalently, a row-major storage format with the Upper triangular part.
  // Otherwise, you might get a 20% slowdown for the full factorization step,
  // and rank-updates can be up to 3 times slower.

  // AtI * G
  H.block<da, dp>(ia, 0).noalias() += ph.ItA.transpose() * G;
  // AtA
  H.block<da, da>(ia, ia) += ph.AtA;
  // Atr
  b.segment<da>(ia) -= ph.Atr;
}

/// ============================================================================
FrameHessian2::FrameHessian2(int fi, int fj) : i_{fi}, j_{fj} {
  CHECK_GE(fi, 0);
  CHECK_GE(fj, 0);
  CHECK_NE(fi, fj);
}

std::string FrameHessian2::Repr() const {
  return fmt::format("FrameHessian2(i={}, j={}, n={}, c={:.4e})", i_, j_, n, c);
}

FrameHessian2& FrameHessian2::operator+=(const FrameHessian2& rhs) {
  CHECK_GE(i_, 0);
  CHECK_GE(j_, 0);
  CHECK_EQ(i_, rhs.i_);
  CHECK_EQ(j_, rhs.j_);

  if (rhs.n > 0) {
    Hii += rhs.Hii;
    Hij += rhs.Hij;
    Hjj += rhs.Hjj;

    bi += rhs.bi;
    bj += rhs.bj;

    c += rhs.c;
    n += rhs.n;
  }

  return *this;
}

void FrameHessian2::AddPatchHess(const PatchHessian2& ph,
                                 const Matrix26d& G0,
                                 const Matrix26d& G1,
                                 int affine_offset) noexcept {
  ++n;

  // ---------------------------------------------------------------------------
  // JtJ                                              | -Jtr
  // ---------------------------------------------------------------------------
  // [Hii,                  , Hij                   ] | [bi]
  // [                      , Hjj                   ] | [bj]
  // ---------------------------------------------------------------------------
  // [[G0t*ItI*G0, G0t*ItA0], [G0t*ItI*G1, G0t*ItA1]] | [G0t*Itr]
  // [[   A0tI*G0,   A0tA0)], [   A0tI*G1,    A0tA1]] | [  A0t*r]
  //                          [G1t*ItI*G1, G1t*ItA1]] | [G1t*Itr]
  //                          [   A1tI*G1,   A1t*A1]] | [  A1t*r]
  // ---------------------------------------------------------------------------
  // Note that we don't strictly enforce i < j, so it is possible when i > j, we
  // have a lower triangular form. This is taken care of in
  // FullBlockHessian.Add()

  // Pose block
  // G0t * ItI * G0
  Hii.topLeftCorner<dp, dp>().noalias() += G0.transpose() * ph.ItI * G0;
  // G0t * ItI * G1
  Hij.topLeftCorner<dp, dp>().noalias() += G0.transpose() * ph.ItI * G1;
  // G1t * ItI * G1
  Hjj.topLeftCorner<dp, dp>().noalias() += G1.transpose() * ph.ItI * G1;

  // G0t * Itr
  bi.head<dp>().noalias() -= G0.transpose() * ph.Itr;
  // G1t * Itr
  bj.head<dp>().noalias() -= G1.transpose() * ph.Itr;

  // cost
  c += ph.r2;

  if (affine_offset < 0) return;
  // since we only project points from left image, ia0 should always be 0
  // ia1 depends on which image we project into, 0 for left, 2 for right
  const int ia0 = dp;
  const int ia1 = dp + affine_offset;

  // We only fill the upper-triangular part for now and make it symmetric later
  // before solving

  // G0t * ItA0
  Hii.block<dp, da>(0, ia0).noalias() += G0.transpose() * ph.ItA0;
  // A0t * A0
  Hii.block<da, da>(ia0, ia0) += ph.A0tA0;

  // This is already an off-diagonal block so we need it to be full
  // G0t * It * A1
  Hij.block<dp, da>(0, ia1).noalias() += G0.transpose() * ph.ItA1;
  // A0t * I * G1
  Hij.block<da, dp>(ia0, 0).noalias() += ph.ItA0.transpose() * G1;
  // A0t * A1
  Hij.block<da, da>(ia0, ia1) += ph.A0tA1;

  // G1t * ItA1
  Hjj.block<dp, da>(ia1, 0).noalias() += G1.transpose() * ph.ItA1;
  // A1t * A1
  Hjj.block<da, da>(ia1, ia1) += ph.A1tA1;

  // A0t * r
  bi.segment<da>(ia0) -= ph.A0tr;
  // A1t * r
  bj.segment<da>(ia1) -= ph.A1tr;
}

/// ============================================================================
void FrameHessianX::ResetData() noexcept {
  Reset();

  if (!data_.empty()) {
    storage().setZero();
  }
}

int FrameHessianX::MapFrames(int nframes) {
  const int dim = nframes * Dim::kFrame;
  const int size = dim * dim + dim;
  ResizeData(size);

  auto* ptr = data_ptr();
  new (&Hpp) MatrixXdMap(ptr, dim, dim);
  ptr += Hpp.size();
  new (&bp) VectorXdMap(ptr, dim);
  ptr += bp.size();

  CHECK_EQ(ptr - data_ptr(), size);
  return size;
}

void FrameHessianX::Scale(double s) noexcept {
  if (s == 1.0) return;
  Hpp.array() *= s;
  bp.array() *= s;
}

void FrameHessianX::AddDiag(int ind, int size, double val) {
  CHECK_GE(ind, 0);
  CHECK_LE(ind + size, Hpp.rows())
      << fmt::format("ind: {}, size: {}, H: {}", ind, size, Hpp.rows());
  Hpp.diagonal().segment(ind, size).array() += val;
}

void FrameHessianX::FixGauge(double lambda, bool fix_scale) {
  CHECK_GE(num_frames(), 1);
  AddDiag(0, df, lambda);

  if (fix_scale) {
    // Fix the translation parameter of the second frame
    // Ideally one only need to fix one direction of translation
    CHECK_GE(num_frames(), 2);
    AddDiag(df + 3, 3, lambda);
  }
}

/// ============================================================================
void SchurFrameHessian::AddPriorHess(const PriorFrameHessian& prior) {
  CHECK(prior.Ok());
  CHECK_GE(num_frames(), prior.num_frames());

  const auto m = prior.dim_frames();
  Hpp.topLeftCorner(m, m) += prior.Hpp;
  bp.head(m) += prior.bp;
  n += prior.n;
}

void SchurFrameHessian::Solve(VectorXdRef xp, VectorXdRef yp) {
  CHECK(Ok());
  CHECK(!empty());
  CHECK_EQ(xp.size(), dim_frames());

  // s is a copy
  const auto s =
      (Hpp.diagonal().array().abs() + 10).sqrt().inverse().matrix().eval();
  const auto S = s.asDiagonal();

  // Use inplace decomposition
  // https://eigen.tuxfamily.org/dox/group__InplaceDecomposition.html
  // For 4 frames, inplace solve is roughly 10x faster than normal solve (1us vs
  // 10us).
  // Because we add both a large and small value to the diagonal, we use ldlt to
  // ensure stability of the decomposition
  LdltLowerInplace solver(Hpp);
  xp = solver.solve(bp);
  yp = S.inverse() * xp;

  // Inplace decomp modifies storage, so we reset n
  n = 0;
}

void SchurFrameHessian::MargFrame(PriorFrameHessian& prior, int fid) {
  CHECK(Ok());
  CHECK_GE(fid, 0);
  CHECK_LT(fid, num_frames());
  CHECK_EQ(num_frames(), prior.num_frames() + 1);

  // First rotate the block that needs to be marginalized to top left. It is
  // possible to also rotate it to bottom right, but it is highly likely that
  // the oldest frame will be marginalized, so move it to top left will take
  // fewer operations (noop if its already there). Therefore, the subsequent
  // frame marginalization operation will be different from the point
  // marginalization one.
  StableRotateBlockTopLeft(Hpp, bp, fid, df);

  // Now we can marginalize the top left block of Hsc on to the rest
  MargTopLeftBlock(Hpp, bp, prior.Hpp, prior.bp, df);

  // Record how many points are marginalized
  prior.n = n;
}

/// ============================================================================
int FramePointHessian::CalcDataSize(int frame_dim, int point_dim) noexcept {
  // Hpp + Hpm + Hmm + bp + bm + xp + xm
  return frame_dim * frame_dim +  // Hpp
         frame_dim * point_dim +  // Hpm
         point_dim +              // Hmm
         frame_dim +              // bp
         point_dim +              // bm
         frame_dim +              // xp
         point_dim;               // xm
}

std::string FramePointHessian::Repr() const {
  return fmt::format(
      "FullBlockHessian(nframes={}, npoints={}, dframes={}, dpoints={}, "
      "num={}, costs={:.4e}, size={}, capacity={}, usage={:.2f}%)",
      num_frames(),
      num_points(),
      dim_frames(),
      dim_points(),
      num_costs(),
      cost(),
      size(),
      capacity(),
      (100.0 * size()) / capacity());
}

auto FramePointHessian::XpAt(int i) const noexcept
    -> Eigen::Map<const Vector10d> {
  CHECK_GE(i, 0);
  CHECK_LT(i, num_frames());
  return Eigen::Map<const Vector10d>(xp.data() + i * df);
}

auto FramePointHessian::XpAt(int i) noexcept -> Eigen::Map<Vector10d> {
  CHECK_GE(i, 0);
  CHECK_LT(i, num_frames());
  return Eigen::Map<Vector10d>(xp.data() + i * df);
}

int FramePointHessian::MapFull(int nframes, int npoints) {
  const int frame_dim = nframes * Dim::kFrame;
  const int point_dim = npoints * Dim::kPoint;
  const int size = CalcDataSize(frame_dim, point_dim);
  ResizeData(size);

  // https://eigen.tuxfamily.org/dox
  // group__TutorialMapClass.html#TutorialMapPlacementNew
  auto* ptr = data_ptr();

  new (&Hpp) MatrixXdMap(ptr, frame_dim, frame_dim);
  ptr += Hpp.size();
  new (&Hpm) MatrixXdMap(ptr, frame_dim, point_dim);
  ptr += Hpm.size();
  new (&Hmm) VectorXdMap(ptr, point_dim);
  new (&Hmm_inv) VectorXdMap(ptr, point_dim);
  ptr += Hmm.size();

  new (&bp) VectorXdMap(ptr, frame_dim);
  ptr += bp.size();
  new (&bm) VectorXdMap(ptr, point_dim);
  ptr += bm.size();

  new (&xp) VectorXdMap(ptr, frame_dim);
  ptr += xp.size();
  new (&xm) VectorXdMap(ptr, point_dim);
  ptr += xm.size();

  CHECK_EQ(ptr - data_ptr(), size);
  return size;
}

void FramePointHessian::ReserveFull(int nframes, int npoints) {
  CHECK_GT(nframes, 0);
  CHECK_GT(npoints, 0);

  const int frame_dim = nframes * Dim::kFrame;
  const int point_dim = npoints * Dim::kPoint;
  const auto size = CalcDataSize(frame_dim, point_dim);
  ReserveData(size);
}

void FramePointHessian::ResetFull(double Hpp_diag) noexcept {
  ready_ = false;
  ResetData();
  if (Hpp_diag > 0) {
    Hpp.diagonal().setConstant(Hpp_diag);
  }
}

void FramePointHessian::AddFrameHess(const FrameHessian2& fh) {
  if (!fh.Ok()) return;
  const int ii = fh.ii();
  const int jj = fh.jj();

  CHECK_GE(ii, 0);
  CHECK_GE(jj, 0);
  CHECK_LT(ii, dim_frames());
  CHECK_LT(jj, dim_frames());
  CHECK_NE(ii, jj);

  // Add each sub block in pose hessian (only upper triangular block)
  Hpp.block<df, df>(ii, ii) += fh.Hii;
  Hpp.block<df, df>(jj, jj) += fh.Hjj;

  if (ii < jj) {
    Hpp.block<df, df>(ii, jj) += fh.Hij;
  } else {
    Hpp.block<df, df>(jj, ii).noalias() += fh.Hij.transpose();
  }

  bp.segment<df>(ii) += fh.bi;
  bp.segment<df>(jj) += fh.bj;

  c += fh.c;
  n += fh.n;
}

void FramePointHessian::AddPatchHess(const PatchHessian2& ph,
                                     const Matrix26d& G0,
                                     const Matrix26d& G1,
                                     const Vector2d& Gd,
                                     const Vector3i& ijh,
                                     int affine_offset) noexcept {
  const int if0 = ijh[0];  // start index of frame 0
  const int if1 = ijh[1];  // start index of frame 1
  const int hid = ijh[2];  // index of depth point

  // G0t * ItI * Gd
  Hpm.block<dp, dd>(if0, hid).noalias() += G0.transpose() * ph.ItI * Gd;
  // G1t * ItI * Gd
  Hpm.block<dp, dd>(if1, hid).noalias() += G1.transpose() * ph.ItI * Gd;

  // Gdt * ItI * Gd
  Hmm[hid] += Gd.transpose() * ph.ItI * Gd;
  // Gdt * Itr
  bm[hid] -= Gd.transpose() * ph.Itr;

  if (affine_offset < 0) return;
  const int ia0 = if0 + dp;  // aff_offset = 0, since it's always left
  const int ia1 = if1 + dp + affine_offset;
  // A0t * I * Gd
  Hpm.block<da, dd>(ia0, hid).noalias() += ph.ItA0.transpose() * Gd;
  // A1t * I * Gd
  Hpm.block<da, dd>(ia1, hid).noalias() += ph.ItA1.transpose() * Gd;
}

void FramePointHessian::Solve() {
  CHECK(ready_);

  // Back-substitute xp to get xm
  xm = Hmm_inv.asDiagonal() * (bm - Hpm.transpose() * xp);
}

void FramePointHessian::Prepare() {
  CHECK(!ready_);
  SafeCwiseInverse(Hmm);  // Hmm_inv
  FillLowerTriangular(Hpp);
  ready_ = true;
}

int FramePointHessian::MargPointsAll(SchurFrameHessian& schur,
                                     int gsize) const {
  CHECK(Ok());
  CHECK(ready_);
  // Hsc = Hpp - Hpm * Hmm^-1 * Hmp
  // bsc = bp  - Hpm * Hmm^-1 * bm
  // Also update n for schur to store number of points marginalized
  const auto n_marged =
      MargPointsToFrames(Hpp, Hpm, Hmm_inv, bp, bm, schur.Hpp, schur.bp, gsize);
  schur.n = n_marged;
  return n_marged;
}

int FramePointHessian::MargPointsRange(SchurFrameHessian& schur,
                                       const cv::Range& range,
                                       int gsize) {
  CHECK(Ok()) << Repr();
  CHECK_GE(range.start, 0);
  CHECK_LT(range.end, dim_points());
  CHECK_GT(range.size(), 0);

  const int i = range.start;
  const int m = range.end - range.start;
  // m stores the number of points being marginalized, which might be different
  // from n, as some points might be considered outlier and have no
  // contribution. Also m is allowed to be 0 since the MargPointsToFramesLower
  // function can handle this case.

  // Manually prepare for marginalization
  FillLowerTriangular(Hpp);
  SafeCwiseInverse(Hmm.segment(i, m));
  const auto n_marged = MargPointsToFrames(Hpp,
                                           Hpm.middleCols(i, m),
                                           Hmm_inv.segment(i, m),
                                           bp,
                                           bm.segment(i, m),
                                           schur.Hpp,
                                           schur.bp,
                                           gsize);
  schur.n = n_marged;
  return n_marged;
}

int MargPointsToFrames(const MatrixXdCRef& Hpp,
                       const MatrixXdCRef& Hpm,
                       const VectorXdCRef& Hmm_inv,
                       const VectorXdCRef& bp,
                       const VectorXdCRef& bm,
                       MatrixXdRef Hsc,
                       VectorXdRef bsc,
                       int gsize) {
  // https://www.robots.ox.ac.uk/~gsibley/Personal/Papers/gsibley-springer2007.pdf
  // [ Hpp Hpm ] [ xp ] = [ bp ]
  // [ Hpm Hmm ] [ xm ] = [ bm ]
  // Hsc = Hpp - Hpm * Hmm^-1 * Hmp
  // bsc = bp  - Hpm * Hmm^-1 * bm

  // Pre-condition
  // 1. Hpp is square/symmetric and matches Hpm and bp
  // 2. Hsc is square and matches Hpp and bsc
  const auto dim_frames = bp.size();
  const auto dim_points = bm.size();
  CHECK_GT(dim_frames, 0);
  CHECK_EQ(Hpp.rows(), dim_frames);
  CHECK_EQ(Hpp.cols(), dim_frames);
  CHECK_EQ(Hpm.rows(), dim_frames);
  CHECK_EQ(Hpm.cols(), dim_points);
  CHECK_EQ(Hmm_inv.size(), dim_points);
  CHECK_EQ(Hpp, Hpp.transpose()) << "\n" << Hpp;

  CHECK_EQ(Hsc.rows(), dim_frames);
  CHECK_EQ(Hsc.cols(), dim_frames);
  CHECK_EQ(bsc.size(), dim_frames);

  // Writing efficient matrix product expressions
  // https://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html
  // m1.noalias() = m4 + m2 * m3;
  // Should be
  // m1 = m4;
  // m1.noalias() += m2 * m3;
  // Simple benchmark shows no huge difference though, probably because Hpp is
  // much smaller compared to Hpm. However, doing this allows us to nicely
  // handle the case where no points are to be marginalized
  Hsc = Hpp;
  bsc = bp;

  // If no points to marg, then just set Hsc to Hpp and return
  if (dim_points == 0) return 0;

  if (gsize <= 0) {
    // Single thread
    Hsc.noalias() -= Hpm * Hmm_inv.asDiagonal() * Hpm.transpose();
  } else {
    // Parallel
    // First compute Hpm * Hmm^-1 * Hpm^T, only fill lower triangular part
    // Hsc is column major, thus we fill each column per thread
    // for each column in lower triangular part we compute
    // Hsc[i:,i] -= Hpm[i:,:] * (Hmm_inv * Hpm[i,:])
    ParallelFor({0, static_cast<int>(dim_frames), gsize}, [&](int i) {
      const auto r = dim_frames - i;
      Hsc.col(i).tail(r).noalias() -=
          Hpm.bottomRows(r) * (Hmm_inv.asDiagonal() * Hpm.row(i).transpose());
    });
  }

  // Make sure Hsc is symmetric
  FillUpperTriangular(Hsc);

  bsc.noalias() -= Hpm * (Hmm_inv.asDiagonal() * bm);

  // Post-condition
  // 1. Hsc shape doesn't change
  CHECK_EQ(Hsc.rows(), dim_frames);
  CHECK_EQ(Hsc.cols(), dim_frames);
  CHECK_EQ(bsc.size(), dim_frames);
  return static_cast<int>((bm.array() != 0).count());
}

int PointHessian::MapPoints(int npoints) {
  const int dim = npoints * Dim::kPoint;
  const int size = dim * 3;
  data_.resize(size);

  auto* ptr = data_.data();
  new (&H) VectorXdMap(ptr, dim);
  ptr += H.size();
  new (&b) VectorXdMap(ptr, dim);
  ptr += b.size();
  new (&x) VectorXdMap(ptr, dim);
  ptr += x.size();

  CHECK_EQ(ptr - data_.data(), size);
  return size;
}

}  // namespace sv::dsol
