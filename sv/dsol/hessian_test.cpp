#include "sv/dsol/hessian.h"

#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>

namespace sv::dsol {
namespace {

namespace bm = benchmark;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/// ============================================================================
TEST(FrameHessianTest, TestDefault) {
  FrameHessian h;
  EXPECT_EQ(h.num_costs(), 0);
  EXPECT_EQ(h.cost(), 0);
  EXPECT_EQ(h.Ok(), false);

  h.n = 1;
  h.c = 2;
  EXPECT_EQ(h.Ok(), true);

  h.Reset();
  EXPECT_EQ(h.num_costs(), 0);
  EXPECT_EQ(h.cost(), 0);
  EXPECT_EQ(h.Ok(), false);
}

/// ============================================================================
TEST(FarmeHessian1Test, TestOpPlus) {
  FrameHessian1 h1;
  h1.n = 1;
  h1.c = 4;

  FrameHessian1 h2;
  h2.n = 3;
  h2.c = 8;

  const auto h12 = h1 + h2;
  EXPECT_EQ(h12.num_costs(), 4);
  EXPECT_EQ(h12.cost(), 12);
}

/// ============================================================================
TEST(FrameHessian2Test, TestBlock2Index) {
  FrameHessian2 h(0, 1);
  EXPECT_EQ(h.ii(), 0);
  EXPECT_EQ(h.jj(), 10);
}

TEST(FrameHessian2Test, TestOpPlus) {
  FrameHessian2 h1(0, 1);
  h1.n = 1;
  h1.c = 4;

  FrameHessian2 h2(0, 1);
  h2.n = 3;
  h2.c = 8;

  const auto h12 = h1 + h2;
  EXPECT_EQ(h12.num_costs(), 4);
  EXPECT_EQ(h12.cost(), 12);
}

/// ============================================================================
TEST(FrameHessianXTest, TestDefault) {
  FrameHessianX h;
  EXPECT_EQ(h.size(), 0);
  EXPECT_EQ(h.empty(), true);
  EXPECT_EQ(h.capacity(), 0);
  EXPECT_EQ(h.storage().size(), 0);
  EXPECT_EQ(h.num_frames(), 0);
  EXPECT_EQ(h.dim_frames(), 0);
}

TEST(FrameHessianXTest, TestMapFrames) {
  FrameHessianX h(2);
  EXPECT_EQ(h.size(), 420);
  EXPECT_EQ(h.empty(), false);
  EXPECT_EQ(h.capacity(), 420);
  EXPECT_EQ(h.storage().size(), 420);
  EXPECT_EQ(h.num_frames(), 2);
  EXPECT_EQ(h.dim_frames(), 20);

  EXPECT_EQ(h.Hpp.rows(), 20);
  EXPECT_EQ(h.Hpp.cols(), 20);
  EXPECT_EQ(h.bp.size(), 20);
}

TEST(FrameHessianXTest, TestSetValues) {
  FrameHessianX h(2);

  EXPECT_EQ(h.Hpp, MatrixXd::Zero(20, 20));
  EXPECT_EQ(h.bp, VectorXd::Zero(20));

  h.Hpp.setConstant(1);
  h.bp.setConstant(2);

  EXPECT_EQ(h.Hpp, MatrixXd::Constant(20, 20, 1));
  EXPECT_EQ(h.bp, VectorXd::Constant(20, 2));

  h.ResetData();
  EXPECT_EQ(h.Hpp, MatrixXd::Zero(20, 20));
  EXPECT_EQ(h.bp, VectorXd::Zero(20));

  h.MapFrames(1);
  EXPECT_EQ(h.Hpp, MatrixXd::Zero(10, 10));
  EXPECT_EQ(h.bp, VectorXd::Zero(10));
}

/// ============================================================================
TEST(SchurFrameHessianTest, TestFixGauge) {
  SchurFrameHessian schur(2);
  EXPECT_EQ(schur.Hpp.diagonal(), VectorXd::Zero(20));

  schur.FixGauge(1, false);
  EXPECT_EQ(schur.Hpp.diagonal().head(10), VectorXd::Ones(10));
  EXPECT_EQ(schur.Hpp.diagonal().tail(10), VectorXd::Zero(10));

  schur.FixGauge(1, true);
  EXPECT_EQ(schur.Hpp.diagonal().head(10), VectorXd::Ones(10) * 2);
  EXPECT_EQ(schur.Hpp.diagonal().segment(10, 3), VectorXd::Zero(3));
  EXPECT_EQ(schur.Hpp.diagonal().segment(13, 3), VectorXd::Ones(3));
}

TEST(SchurFrameHessianTest, TestAddPriorHess) {
  SchurFrameHessian schur(2);
  PriorFrameHessian prior(1);

  prior.Hpp.diagonal().setOnes();
  prior.bp.setZero();
  prior.n = 1;

  schur.AddPriorHess(prior);
  EXPECT_EQ(schur.n, prior.n);
  EXPECT_EQ(schur.Hpp.diagonal().head(10), VectorXd::Ones(10));
  EXPECT_EQ(schur.Hpp.diagonal().tail(10), VectorXd::Zero(10));
}

/// ============================================================================
TEST(FramePointHessianTest, TestReserveFull) {
  FramePointHessian h;
  h.ReserveFull(2, 10);

  EXPECT_EQ(h.dim_full(), 0);
  EXPECT_EQ(h.num_frames(), 0);
  EXPECT_EQ(h.num_points(), 0);
  EXPECT_EQ(h.capacity(), 670);
  EXPECT_EQ(h.size(), 0);
  EXPECT_EQ(h.num_costs(), 0);
}

TEST(FramePointHessianTest, TestMapFull) {
  FramePointHessian h;
  h.MapFull(2, 10);

  EXPECT_EQ(h.dim_full(), 30);
  EXPECT_EQ(h.num_frames(), 2);
  EXPECT_EQ(h.num_points(), 10);
  EXPECT_EQ(h.dim_frames(), 20);
  EXPECT_EQ(h.dim_points(), 10);
  EXPECT_EQ(h.size(), 670);
  EXPECT_EQ(h.num_costs(), 0);
}

TEST(FramePointHessianTest, TestMap) {
  FramePointHessian fh(2, 10);
  fh.Hpp.setOnes();
  fh.Hpm.setOnes();
  fh.Hmm_inv.setOnes();
  fh.bp.setOnes();
  fh.bm.setOnes();
  fh.xp.setOnes();
  fh.xm.setOnes();

  EXPECT_TRUE((fh.storage().array() == 1).all());

  fh.Hpp.setConstant(1);
  fh.Hpm.setConstant(2);
  fh.Hmm_inv.setConstant(3);
  fh.bp.setConstant(4);
  fh.bm.setConstant(5);
  fh.xp.setConstant(6);
  fh.xm.setConstant(7);

  EXPECT_TRUE((fh.Hpp.array() == 1).all());
  EXPECT_TRUE((fh.Hpm.array() == 2).all());
  EXPECT_TRUE((fh.Hmm_inv.array() == 3).all());
  EXPECT_TRUE((fh.bp.array() == 4).all());
  EXPECT_TRUE((fh.bm.array() == 5).all());
  EXPECT_TRUE((fh.xp.array() == 6).all());
  EXPECT_TRUE((fh.xm.array() == 7).all());
}

TEST(FramePointHessianTest, TestAddFrameHessian2) {
  FrameHessian2 h2(0, 1);
  h2.n = 10;
  h2.c = 20;
  h2.Hii.setConstant(1);
  h2.Hij.setConstant(2);
  h2.Hjj.setConstant(3);
  h2.bi.setConstant(4);
  h2.bj.setConstant(5);

  FramePointHessian hess(2, 10);

  hess.AddFrameHess(h2);
  EXPECT_EQ(hess.num_costs(), 10);
  EXPECT_EQ(hess.Hpp(0, 0), 1);
  EXPECT_EQ(hess.Hpp(5, 5), 1);
  EXPECT_EQ(hess.bp(0), 4);
  EXPECT_EQ(hess.bp(5), 4);
  EXPECT_EQ(hess.num_costs(), h2.num_costs());
  EXPECT_EQ(hess.cost(), h2.cost());
}

TEST(FramePointHessianTest, TestMargPointsToFrames) {
  constexpr int P{10};
  constexpr int M{1000};

  MatrixXd A = MatrixXd::Random(P, M);
  MatrixXd Hpp = A * A.transpose();
  MakeSymmetric(Hpp);

  MatrixXd Hpm = MatrixXd::Random(P, M);
  VectorXd Hmm_inv = VectorXd::Ones(M);
  VectorXd bp = VectorXd::Random(P);
  VectorXd bm = VectorXd::Random(M);

  MatrixXd Hsc = MatrixXd::Zero(P, P);
  MatrixXd Hsc0 = MatrixXd::Zero(P, P);
  MatrixXd Hsc1 = MatrixXd::Zero(P, P);
  VectorXd bsc0 = VectorXd::Zero(P);
  VectorXd bsc1 = VectorXd::Zero(P);

  MargPointsToFrames(Hpp, Hpm, Hmm_inv, bp, bm, Hsc, bsc0, 0);
  MargPointsToFrames(Hpp, Hpm, Hmm_inv, bp, bm, Hsc, bsc1, 1);

  EXPECT_TRUE(Hsc0.isApprox(Hsc1)) << (Hsc0 - Hsc1).norm();
  EXPECT_EQ(bsc0, bsc1) << (bsc0 - bsc1).norm();
}

/// ============================================================================
TEST(MargTest, TestMargPointsToFrameLower) {
  MatrixXd H = MatrixXd::Zero(5, 5);
  H.topLeftCorner<2, 2>().setIdentity();
  H.topRightCorner<2, 3>().setOnes();
  H.bottomRightCorner<3, 3>().setIdentity();

  VectorXd b = VectorXd::Ones(5);
  MatrixXd Hsc = MatrixXd::Zero(2, 2);
  VectorXd bsc = VectorXd::Zero(2);

  MargPointsToFrames(H.topLeftCorner<2, 2>(),
                     H.topRightCorner<2, 3>(),
                     H.bottomRightCorner<3, 3>().diagonal(),
                     b.head<2>(),
                     b.tail<3>(),
                     Hsc,
                     bsc,
                     0);
  MatrixXd Hsc0(2, 2);
  Hsc0 << -2, -3, -3, -2;
  VectorXd bsc0(2);
  bsc0 << -2, -2;
  EXPECT_EQ(Hsc, Hsc0);
  EXPECT_EQ(bsc, bsc0);

  // Reset and tbb
  Hsc.setZero();
  bsc.setZero();
  MargPointsToFrames(H.topLeftCorner<2, 2>(),
                     H.topRightCorner<2, 3>(),
                     H.bottomRightCorner<3, 3>().diagonal(),
                     b.head<2>(),
                     b.tail<3>(),
                     Hsc,
                     bsc,
                     1);
  FillUpperTriangular(Hsc);
  EXPECT_EQ(Hsc, Hsc0);
  EXPECT_EQ(bsc, bsc0);
}
/// ============================================================================
void BM_ItIMatrix(bm::State& state) {
  Eigen::Matrix2d ItI;
  PatchHessian::Matrix2Kd It;
  PatchHessian::ArrayKd w;

  for (auto _ : state) {
    ItI = It * w.matrix().asDiagonal() * It.transpose();
    bm::DoNotOptimize(ItI);
  }
}
BENCHMARK(BM_ItIMatrix);

void BM_ItIArray(bm::State& state) {
  Eigen::Matrix2d ItI;
  PatchHessian::Matrix2Kd It;
  PatchHessian::ArrayKd w;

  for (auto _ : state) {
    ItI = It * (It.transpose().array().colwise() * w).matrix();
    bm::DoNotOptimize(ItI);
  }
}
BENCHMARK(BM_ItIArray);

PatchHessian::Matrix2Kd It{PatchHessian::Matrix2Kd::Random()};
PatchHessian::Matrix2Kd At{PatchHessian::Matrix2Kd::Random()};
PatchHessian::ArrayKd w{PatchHessian::ArrayKd::Random()};
PatchHessian::ArrayKd r{PatchHessian::ArrayKd::Random()};

/// ============================================================================
void BM_PatchHessAddI(bm::State& state) {
  PatchHessian ph;

  for (auto _ : state) {
    for (int i = 0; i < It.cols(); ++i) {
      ph.AddI(It.col(i), r[i], w[i]);
    }
    bm::DoNotOptimize(ph);
  }
}
BENCHMARK(BM_PatchHessAddI);

void BM_PatchHessSetI(bm::State& state) {
  PatchHessian ph;

  for (auto _ : state) {
    ph.SetI(It, r, w);
    bm::DoNotOptimize(ph);
  }
}
BENCHMARK(BM_PatchHessSetI);

void BM_PatchHess1AddA(bm::State& state) {
  PatchHessian1 ph;

  for (auto _ : state) {
    for (int i = 0; i < It.cols(); ++i) {
      ph.AddA(It.col(i), At.col(i), r[i], w[i]);
    }
    bm::DoNotOptimize(ph);
  }
}
BENCHMARK(BM_PatchHess1AddA);

void BM_PatchHess1SetA(bm::State& state) {
  PatchHessian1 ph;

  for (auto _ : state) {
    ph.SetA(It, At, r, w);
    bm::DoNotOptimize(ph);
  }
}
BENCHMARK(BM_PatchHess1SetA);

void BM_PatchHess2AddA(bm::State& state) {
  PatchHessian2 ph;

  for (auto _ : state) {
    for (int i = 0; i < It.cols(); ++i) {
      ph.AddA(It.col(i), At.col(i), At.col(i), r[i], w[i]);
    }
    bm::DoNotOptimize(ph);
  }
}
BENCHMARK(BM_PatchHess2AddA);

void BM_PatchHess2SetA(bm::State& state) {
  PatchHessian2 ph;

  for (auto _ : state) {
    ph.SetA(It, At, At, r, w);
    bm::DoNotOptimize(ph);
  }
}
BENCHMARK(BM_PatchHess2SetA);

/// ============================================================================
constexpr int kFrames = 4;
constexpr int kPoints = 1000;

void BM_MargPointsToFramesLower(bm::State& state) {
  const auto dp = kFrames * state.range(0);
  const auto dm = kFrames * kPoints;
  MatrixXd Hpp = MatrixXd::Identity(dp, dp);
  VectorXd bp = VectorXd::Zero(dp);
  MatrixXd Hpm = MatrixXd::Zero(dp, dm);
  VectorXd Hmm_inv = VectorXd::Ones(dm);
  VectorXd bm = VectorXd::Zero(dm);

  MatrixXd Hsc = MatrixXd::Zero(dp, dp);
  VectorXd bsc = VectorXd::Zero(dp);
  for (auto _ : state) {
    MargPointsToFrames(Hpp, Hpm, Hmm_inv, bp, bm, Hsc, bsc);
    bm::DoNotOptimize(Hsc);
  }
}
BENCHMARK(BM_MargPointsToFramesLower)->Arg(6)->Arg(8)->Arg(10);

void BM_FramePointHessMargPointsAll(bm::State& state) {
  SchurFrameHessian schur(kFrames);
  FramePointHessian block(kFrames, kFrames * kPoints);
  block.n = 1;
  block.Hpp.setIdentity();
  block.Hpm.setZero();
  block.Hmm_inv.setOnes();
  block.Prepare();

  const auto gsize = static_cast<int>(state.range(0));
  for (auto _ : state) {
    block.MargPointsAll(schur, gsize);
  }
}
BENCHMARK(BM_FramePointHessMargPointsAll)->Arg(0)->Arg(1);

void BM_FullHessMargPointsRange(bm::State& state) {
  SchurFrameHessian schur(kFrames);
  FramePointHessian block(kFrames, kFrames * kPoints);
  block.n = 1;
  block.Hpp.setIdentity();
  block.Hpm.setZero();
  block.Hmm_inv.setOnes();
  block.Prepare();

  const auto gsize = static_cast<int>(state.range(0));
  for (auto _ : state) {
    block.MargPointsRange(schur, {0, kPoints}, gsize);
  }
}
BENCHMARK(BM_FullHessMargPointsRange)->Arg(0)->Arg(1);

/// ============================================================================
// The following benchmark shows that simply calling inverse is faster then LLT
// for marginalizing 1 or 2 frames, but slower when marginalizing 3 frames.
// Since we only marginalize one frame at a time, we will go with the inverse
constexpr int N = 50;
void BM_MargFrameInverse(bm::State& state) {
  const auto m = state.range(0);
  Eigen::MatrixXd H00(m, m);
  Eigen::MatrixXd H01(m, N - m);
  Eigen::MatrixXd H11(N - m, N - m);
  H00.setIdentity();
  H01.setOnes();

  for (auto _ : state) {
    H11.noalias() = H01.transpose() * H00.inverse() * H01;
    bm::DoNotOptimize(H11);
  }
}
BENCHMARK(BM_MargFrameInverse)->Arg(10)->Arg(20)->Arg(30);

void BM_MargFrameLLT(bm::State& state) {
  const auto m = state.range(0);
  Eigen::MatrixXd H00(m, m);
  Eigen::MatrixXd H01(m, N - m);
  Eigen::MatrixXd H11(N - m, N - m);
  H00.setIdentity();
  H01.setOnes();

  for (auto _ : state) {
    H11.noalias() = H01.transpose() *
                    (H00.selfadjointView<Eigen::Lower>().llt().solve(H01));
    bm::DoNotOptimize(H11);
  }
}
BENCHMARK(BM_MargFrameLLT)->Arg(10)->Arg(20)->Arg(30);

}  // namespace
}  // namespace sv::dsol
