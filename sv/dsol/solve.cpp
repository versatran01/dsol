#include "sv/dsol/solve.h"

#include <Eigen/Cholesky>

#include "sv/util/logging.h"

namespace sv::dsol {

void SolveCholesky(const MatrixXdCRef& A,
                   const VectorXdCRef& b,
                   VectorXdRef x) {
  const auto n = x.size();
  CHECK_EQ(A.rows(), n);
  CHECK_EQ(A.rows(), n);
  CHECK_EQ(b.size(), n);

  x = A.selfadjointView<Eigen::Lower>().llt().solve(b);
}

void SolveCholeskyScaled(const MatrixXdCRef& A,
                         const VectorXdCRef& b,
                         VectorXdRef x,
                         VectorXdRef xs) {
  CHECK_EQ(x.size(), xs.size());

  // Scaling for better numerical stability
  // See
  // Numerical Methods in Matrix Computations, by Ake Bjorck
  // Use L1 norm
  // S = 1 / sqrt(|diag(H)|_p + 10)
  const auto s =
      (A.diagonal().array().abs() + 10).sqrt().inverse().matrix().eval();
  const auto S = s.asDiagonal();
  // Note that since S is diagonal, we can safely multiply S * A * S

  // As = S * A * S
  // bs = S * b
  // Solve As * xs = bs
  // S*A*S * xs = S*b => A * (S*xs) = b
  // Then x = S * xs

  // From Bjork Theorem 1.2.7
  // "It is important to realize that employing an optimal row or column scaling
  // may not improve the computed solution. Indeed, for a fixed pivot sequence,
  // the solution computed by GE is not affected by such scaling."
  // Nevertheless, we still scale it to determine whether to stop early or not
  SolveCholesky(S * A * S, S * b, xs);
  x = S * xs;
}

}  // namespace sv::dsol
