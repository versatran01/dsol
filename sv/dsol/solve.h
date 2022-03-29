#pragma once

#include "sv/util/eigen.h"

namespace sv::dsol {

/// @brief Simple cholesky solve using LLT, assumes lower triangular
void SolveCholesky(const MatrixXdCRef& A, const VectorXdCRef& b, VectorXdRef x);
/// @brief Same as above, but use row and column scaling and returns the scaled
/// solution as xs
void SolveCholeskyScaled(const MatrixXdCRef& A,
                         const VectorXdCRef& b,
                         VectorXdRef x,
                         VectorXdRef xs);

}  // namespace sv::dsol
