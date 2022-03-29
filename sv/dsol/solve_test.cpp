#include "sv/dsol/solve.h"

#include <fmt/ostream.h>
#include <gtest/gtest.h>

#include "sv/util/logging.h"

namespace sv::dsol {
namespace {

struct System {
  Eigen::MatrixXd A;
  Eigen::VectorXd x;
  Eigen::VectorXd b;
};

System MakeTestSystem(int dim, int num_obs) {
  System sys;

  Eigen::MatrixXd J = Eigen::MatrixXd::Random(num_obs, dim);
  sys.A = J.transpose() * J;
  sys.x = Eigen::VectorXd::Random(dim);
  sys.b = sys.A * sys.x;

  return sys;
}

TEST(SolveTest, TestSolveCholesky) {
  int n = 10;
  int m = 100;
  const auto sys = MakeTestSystem(n, m);

  Eigen::VectorXd x(n);
  SolveCholesky(sys.A, sys.b, x);
  EXPECT_TRUE(x.isApprox(sys.x)) << fmt::format(
      "\n x_true: {}\n x_solved: {}", sys.x.transpose(), x.transpose());
}

TEST(SolveTest, TestSolveCholeskyScaled) {
  int n = 10;
  int m = 100;
  const auto sys = MakeTestSystem(n, m);

  Eigen::VectorXd x(n);
  Eigen::VectorXd v(n);
  SolveCholeskyScaled(sys.A, sys.b, x, v);
  EXPECT_TRUE(x.isApprox(sys.x)) << fmt::format(
      "\n x_true: {}\n x_solved: {}", sys.x.transpose(), x.transpose());
}

}  // namespace
}  // namespace sv::dsol
