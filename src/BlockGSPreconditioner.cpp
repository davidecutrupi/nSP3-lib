#include "BlockGSPreconditioner.hpp"
#include "GlobalTimer.hpp"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>

namespace solver {
  using namespace dealii;

  template <typename BlockVectorType, typename Inverse00Type, typename Inverse22Type, typename Operator20Type>
  void BlockGSPreconditioner<BlockVectorType, Inverse00Type, Inverse22Type, Operator20Type>::clear() {
    tmp_A20_dst0.reinit(0);
    tmp_rhs_2.reinit(0);
    tmp_A20T_dst2.reinit(0);
    tmp_rhs_0.reinit(0);
  }


  template <typename BlockVectorType, typename Inverse00Type, typename Inverse22Type, typename Operator20Type>
  void BlockGSPreconditioner<BlockVectorType, Inverse00Type, Inverse22Type, Operator20Type>::initialize(const BlockVectorType &template_vector) {
    clear();
    
    // Reinit temp vectors with the same size of U2
    tmp_A20_dst0.reinit(template_vector.block(1));
    tmp_rhs_2.reinit(template_vector.block(1));

    // Reinit temp vectors with the same size of U0
    tmp_A20T_dst2.reinit(template_vector.block(0));
    tmp_rhs_0.reinit(template_vector.block(0));
  }


  template <typename BlockVectorType, typename Inverse00Type, typename Inverse22Type, typename Operator20Type>
  void BlockGSPreconditioner<BlockVectorType, Inverse00Type, Inverse22Type, Operator20Type>::vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    Assert(dst.n_blocks() == 2 && src.n_blocks() == 2, ExcInternalError());
    
    auto &dst0 = dst.block(0);
    auto &dst2 = dst.block(1);
    const auto &src0 = src.block(0);
    const auto &src2 = src.block(1);

    // 1) Perform A00_sym ^-1 * src0 and find an approx of U0
    {
      TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::BlockGS::InverseZero");
      op_invA00->vmult(dst0, src0);
    }

    // 2) Perform A20 * U0 and find the contribution given by U0 to U2
    {
      TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::BlockGS::Coupling");
      op_A20->vmult(tmp_A20_dst0, dst0);
    }

    // 3) Construct the RHS of for the second step of the preconditioner: src2 minus the contribution given by U0
    tmp_rhs_2 = src2;
    tmp_rhs_2.add(-1.0, tmp_A20_dst0);

    // 4) Perform A22 ^-1 * RHS and find an approx of U2
    {
      TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::BlockGS::InverseSecond");
      op_invA22->vmult(dst2, tmp_rhs_2);
    }
  }


  template <typename BlockVectorType, typename Inverse00Type, typename Inverse22Type, typename Operator20Type>
  void BlockGSPreconditioner<BlockVectorType, Inverse00Type, Inverse22Type, Operator20Type>::Tvmult(BlockVectorType &dst, const BlockVectorType &src) const {
    Assert(dst.n_blocks() == 2 && src.n_blocks() == 2, ExcInternalError());
    
    auto &dst0 = dst.block(0);
    auto &dst2 = dst.block(1);
    const auto &src0 = src.block(0);
    const auto &src2 = src.block(1);
    
    // 1) Perform A22 ^-1 * src2 and find an approx of U2
    {
      TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::BlockGS::InverseZero");
      op_invA22->Tvmult(dst2, src2);
    }

    // 2) Perform A20 * U2 and find the contribution given by U2 to U0
    {
      TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::BlockGS::Coupling");
      op_A20->Tvmult(tmp_A20T_dst2, dst2);
    }

    // 3) Construct the RHS of for the second step of the preconditioner: src0 minus the contribution given by U2
    tmp_rhs_0 = src0;
    tmp_rhs_0.add(-1.0, tmp_A20T_dst2);
    
    // 4) Perform A00_sym ^-1 * RHS and find an approx of U0
    {
      TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::BlockGS::InverseSecond");
      op_invA00->Tvmult(dst0, tmp_rhs_0);
    }

  }

}


#include "InverseOperator.hpp"
#include "MultigridPreconditioner.hpp"
#include "ZeroModeOperator.hpp"
#include "SecondModeOperator.hpp"
#include "CouplingOperator.hpp"
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>

template class solver::BlockGSPreconditioner<
  dealii::LinearAlgebra::distributed::BlockVector<double>,
  solver::InverseOperator<
    dealii::LinearAlgebra::distributed::Vector<double>,
    solver::ZeroModeOperator<3u, double>,
    solver::MultigridPreconditioner<3u, float, solver::ZeroModeOperator<3u, float>>
  >,
  solver::InverseOperator<
    dealii::LinearAlgebra::distributed::Vector<double>,
    solver::SecondModeOperator<3u, double>,
    dealii::PreconditionChebyshev<solver::SecondModeOperator<3u, double>, dealii::LinearAlgebra::distributed::Vector<double>>
  >,
  solver::CouplingOperator<3u, double>
>;

template class solver::BlockGSPreconditioner<
  dealii::LinearAlgebra::distributed::BlockVector<double>,
  solver::InverseOperator<
    dealii::LinearAlgebra::distributed::Vector<double>,
    solver::ZeroModeOperator<2u, double>,
    solver::MultigridPreconditioner<2u, float, solver::ZeroModeOperator<2u, float>>
  >,
  solver::InverseOperator<
    dealii::LinearAlgebra::distributed::Vector<double>,
    solver::SecondModeOperator<2u, double>,
    dealii::PreconditionChebyshev<solver::SecondModeOperator<2u, double>, dealii::LinearAlgebra::distributed::Vector<double>>
  >,
  solver::CouplingOperator<2u, double>
>;


template class solver::BlockGSPreconditioner<
  dealii::LinearAlgebra::distributed::BlockVector<double>,
  solver::InverseOperator<
    dealii::LinearAlgebra::distributed::Vector<double>,
    solver::ZeroModeOperator<3u, double>,
    solver::MultigridPreconditioner<3u, float, solver::ZeroModeOperator<3u, float>>
  >,
  dealii::PreconditionChebyshev<solver::SecondModeOperator<3u, double>, dealii::LinearAlgebra::distributed::Vector<double>>,
  solver::CouplingOperator<3u, double>
>;

template class solver::BlockGSPreconditioner<
  dealii::LinearAlgebra::distributed::BlockVector<double>,
  solver::InverseOperator<
    dealii::LinearAlgebra::distributed::Vector<double>,
    solver::ZeroModeOperator<2u, double>,
    solver::MultigridPreconditioner<2u, float, solver::ZeroModeOperator<2u, float>>
  >,
  dealii::PreconditionChebyshev<solver::SecondModeOperator<2u, double>, dealii::LinearAlgebra::distributed::Vector<double>>,
  solver::CouplingOperator<2u, double>
>;


template class solver::BlockGSPreconditioner<
  dealii::LinearAlgebra::distributed::BlockVector<double>,
  solver::InverseOperator<
    dealii::LinearAlgebra::distributed::Vector<double>,
    solver::ZeroModeOperator<3u, double>,
    solver::MultigridPreconditioner<3u, float, solver::ZeroModeOperator<3u, float>>
  >,
  solver::InverseOperator<
    dealii::LinearAlgebra::distributed::Vector<double>,
    solver::SecondModeOperator<3u, double>,
    solver::MultigridPreconditioner<3u, float, solver::SecondModeOperator<3u, float>>
  >,
  solver::CouplingOperator<3u, double>
>;

template class solver::BlockGSPreconditioner<
  dealii::LinearAlgebra::distributed::BlockVector<double>,
  solver::InverseOperator<
    dealii::LinearAlgebra::distributed::Vector<double>,
    solver::ZeroModeOperator<2u, double>,
    solver::MultigridPreconditioner<2u, float, solver::ZeroModeOperator<2u, float>>
  >,
  solver::InverseOperator<
    dealii::LinearAlgebra::distributed::Vector<double>,
    solver::SecondModeOperator<2u, double>,
    solver::MultigridPreconditioner<2u, float, solver::SecondModeOperator<2u, float>>
  >,
  solver::CouplingOperator<2u, double>
>;