#include "sv/util/eigen.h"

#include <Eigen/Cholesky>

#include "sv/util/logging.h"

namespace sv {

void StableRotateBlockTopLeft(MatrixXdRef H,
                              VectorXdRef b,
                              int block_ind,
                              int block_size) {
  CHECK_EQ(H.rows(), b.size());
  CHECK_EQ(H.cols(), b.size());
  CHECK_GE(block_ind, 0);
  CHECK_GT(block_size, 0);
  CHECK_LT(block_ind * block_size, H.rows());

  if (block_ind == 0) return;
  const auto n = block_size;

  // Permute block of rows up gradually, also need to permute b
  for (int i = block_ind; i > 0; --i) {
    const auto r = i * block_size;
    // swap current rows with above rows
    H.middleRows(r, n).swap(H.middleRows(r - n, n));
    b.segment(r, n).swap(b.segment(r - n, n));
  }

  // Permute block of cols
  for (int j = block_ind; j > 0; --j) {
    const auto c = j * block_size;
    // swap current cols with left cols
    H.middleCols(c, n).swap(H.middleCols(c - n, n));
  }
}

void FillLowerTriangular(MatrixXdRef M) {
  CHECK_EQ(M.rows(), M.cols());
  M.triangularView<Eigen::Lower>() =
      M.triangularView<Eigen::Upper>().transpose();
}

void FillUpperTriangular(MatrixXdRef M) {
  CHECK_EQ(M.rows(), M.cols());
  M.triangularView<Eigen::Upper>() =
      M.triangularView<Eigen::Lower>().transpose();
}

void MakeSymmetric(MatrixXdRef M) {
  CHECK_EQ(M.rows(), M.cols());
  M += M.transpose().eval();
  M.array() /= 2.0;
}

void MargTopLeftBlock(const MatrixXdCRef& Hf,
                      const VectorXdCRef& bf,
                      MatrixXdRef Hm,
                      VectorXdRef bm,
                      int dim) {
  // Pre-condition
  // 1. Hf is square and match bsc and symmetric
  // 2. Hm is quare and match bpr
  const auto nf = bf.size();
  const auto nm = bm.size();
  CHECK_GT(dim, 0);
  CHECK_EQ(nm + dim, nf);
  CHECK_EQ(Hf.rows(), nf);
  CHECK_EQ(Hf.cols(), nf);
  CHECK_EQ(Hm.rows(), nm);
  CHECK_EQ(Hm.cols(), nm);
  CHECK_EQ(Hf, Hf.transpose()) << "\n" << Hf;

  // Hf                     bf
  // [ H00 H01 ] [ x0 ] = [ b0 ]
  // [ H10 H11 ] [ x1 ] = [ b1 ]
  // Hm = H11 - H10 * H00^-1 * H01
  // bm =  b1 - H10 * H00^-1 * b0
  const auto H01 = Hf.topRightCorner(dim, nm);
  const auto H10 = Hf.bottomLeftCorner(nm, dim);

  // Benchmark shows that simply inverse has similar speed as llt
  // However to account for rank-deficiency we use ldlt to inverse
  const auto H00_inv = Hf.topLeftCorner(dim, dim)
                           .selfadjointView<Eigen::Lower>()
                           .ldlt()
                           .solve(Eigen::MatrixXd::Identity(dim, dim))
                           .eval();

  Hm = Hf.bottomRightCorner(nm, nm);
  Hm.noalias() -= H10 * H00_inv * H01;

  const auto b0 = bf.head(dim);
  bm = bf.tail(nm);  // b1
  bm.noalias() -= H10 * (H00_inv * b0);

  // Make sure Hpr is symmetric
  MakeSymmetric(Hm);

  // Post-condition
  // 1. Hpr shape doesn't change
  // 2. Hpr is symmetric
  CHECK_EQ(Hm.rows(), nm);
  CHECK_EQ(Hm.cols(), nm);
  CHECK_EQ(Hm, Hm.transpose()) << "\n" << Hm;
}

}  // namespace sv
