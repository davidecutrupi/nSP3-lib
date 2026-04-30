#include "BlockDiagonalPreconditioner.hpp"

namespace solver {

  template <typename BlockVectorType, typename InnerPreconditionerZero, typename InnerPreconditionerSecond>
  void BlockDiagonalPreconditioner<BlockVectorType, InnerPreconditionerZero, InnerPreconditionerSecond>::vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    prec_zero->vmult(dst.block(0), src.block(0));
    prec_second->vmult(dst.block(1), src.block(1));
  }

  template <typename BlockVectorType, typename InnerPreconditionerZero, typename InnerPreconditionerSecond>
  void BlockDiagonalPreconditioner<BlockVectorType, InnerPreconditionerZero, InnerPreconditionerSecond>::Tvmult(BlockVectorType &dst, const BlockVectorType &src) const {
    prec_zero->Tvmult(dst.block(0), src.block(0));
    prec_second->Tvmult(dst.block(1), src.block(1));
  }
  
}


#include "MultigridPreconditioner.hpp"
#include "ZeroModeOperator.hpp"
#include "SecondModeOperator.hpp"
#include <deal.II/lac/la_parallel_block_vector.h>

template class solver::BlockDiagonalPreconditioner<
  dealii::LinearAlgebra::distributed::BlockVector<double>,
  solver::MultigridPreconditioner<3u, float, solver::ZeroModeOperator<3u, float>>,
  solver::MultigridPreconditioner<3u, float, solver::SecondModeOperator<3u, float>>
>;

template class solver::BlockDiagonalPreconditioner<
  dealii::LinearAlgebra::distributed::BlockVector<double>,
  solver::MultigridPreconditioner<2u, float, solver::ZeroModeOperator<2u, float>>,
  solver::MultigridPreconditioner<2u, float, solver::SecondModeOperator<2u, float>>
>;

template class solver::BlockDiagonalPreconditioner<
  dealii::LinearAlgebra::distributed::BlockVector<double>,
  solver::MultigridPreconditioner<1u, float, solver::ZeroModeOperator<1u, float>>,
  solver::MultigridPreconditioner<1u, float, solver::SecondModeOperator<1u, float>>
>;