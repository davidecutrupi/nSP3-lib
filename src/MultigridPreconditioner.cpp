#include "MultigridPreconditioner.hpp"
#include "GlobalTimer.hpp"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/base/index_set.h>

#include <deal.II/multigrid/mg_tools.h>


namespace solver {
  using namespace dealii;

  template <
    unsigned int dim,
    typename number,
    typename OperatorType,
    typename VectorType,
    typename TransferType,
    typename SmootherPreconditionerType,
    typename CoarseWrapperType>
  void MultigridPreconditioner<dim, number, OperatorType, VectorType, TransferType, SmootherPreconditionerType, CoarseWrapperType>::clear() {
    preconditioner.reset();
    multigrid.reset();

    mg_coarse_wrapper.reset();
    coarse_matrix.reset();

    mg_smoother_wrapper.reset();
    level_smoothers.reset();

    mg_matrix_wrapper.reset();

    level_dof_handlers.clear();
    transfer.reset();
  }


  template <
    unsigned int dim,
    typename number,
    typename OperatorType,
    typename VectorType,
    typename TransferType,
    typename SmootherPreconditionerType,
    typename CoarseWrapperType>
  void MultigridPreconditioner<dim, number, OperatorType, VectorType, TransferType, SmootherPreconditionerType, CoarseWrapperType>::initialize(const DoFHandler<dim> &dof_handler, const std::vector<std::shared_ptr<OperatorType>> &level_operators, const std::vector<std::shared_ptr<DoFHandler<dim>>> &level_dof_handlers, std::shared_ptr<TransferType> mg_transfer, const AffineConstraints<number> &coarse_constraints, const CoarseSolverPolicy &coarse_solver_policy) {
    clear();

    this->transfer = mg_transfer;
    this->level_dof_handlers = level_dof_handlers;

    const unsigned int nlevels = level_operators.size();

    mg_matrix_wrapper = std::make_unique<LevelMatrixWrapper<VectorType, OperatorType>>(level_operators);
    level_smoothers = std::make_unique<MGLevelObject<SmootherType>>(0, nlevels - 1);

    // Setup smoother data
    {
      TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::Setup::Multigrid::Smoother");
      for (unsigned int level = 1; level < nlevels; ++level) {
        typename SmootherType::AdditionalData smoother_data;
        smoother_data.smoothing_range = 15.;
        smoother_data.degree = 5;
        smoother_data.eig_cg_n_iterations = 10;
      
        level_operators[level]->compute_diagonal();
        smoother_data.preconditioner = level_operators[level]->get_matrix_diagonal_inverse();
  
        (*level_smoothers)[level].initialize(*level_operators[level], smoother_data);
      }
    }

    // Initialize mg_smoother
    mg_smoother_wrapper = std::make_unique<LevelSmootherWrapper<VectorType, SmootherType>>();
    mg_smoother_wrapper->initialize(level_smoothers.get());
    
    // Assemble coarse matrix
    {
      TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::Setup::Multigrid::CoarseMatrix");
      coarse_matrix = std::make_unique<TrilinosWrappers::SparseMatrix>();
      CoarseMatrixBuilder<dim, number, OperatorType>::build(*this->level_dof_handlers[0], *level_operators[0], *coarse_matrix, coarse_constraints);
    }

    // Initialize direct solver
    {
      TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::Setup::Multigrid::DirectSolver");
      mg_coarse_wrapper = std::make_unique<CoarseWrapperType>();
      mg_coarse_wrapper->initialize(*coarse_matrix, *this->level_dof_handlers[0], coarse_solver_policy);
    }

    // In the end setup multigrid and precondtioner
    multigrid = std::make_unique<Multigrid<VectorType>>(*mg_matrix_wrapper, *mg_coarse_wrapper, *this->transfer, *mg_smoother_wrapper, *mg_smoother_wrapper);
    preconditioner = std::make_unique<PreconditionMG<dim, VectorType, TransferType>>(dof_handler, *multigrid, *this->transfer);
  }


  template <
    unsigned int dim,
    typename number,
    typename OperatorType,
    typename VectorType,
    typename TransferType,
    typename SmootherPreconditionerType,
    typename CoarseWrapperType>
  template <typename OtherVectorType>
  void MultigridPreconditioner<dim, number, OperatorType, VectorType, TransferType, SmootherPreconditionerType, CoarseWrapperType>::vmult(OtherVectorType &dst, const OtherVectorType &src) const {
    preconditioner->vmult(dst, src);
  }


  template <
    unsigned int dim,
    typename number,
    typename OperatorType,
    typename VectorType,
    typename TransferType,
    typename SmootherPreconditionerType,
    typename CoarseWrapperType>
  template <typename OtherVectorType>
  void MultigridPreconditioner<dim, number, OperatorType, VectorType, TransferType, SmootherPreconditionerType, CoarseWrapperType>::Tvmult(OtherVectorType &dst, const OtherVectorType &src) const {
    preconditioner->Tvmult(dst, src);
  }

}




#include "ZeroModeOperator.hpp"
#include "SecondModeOperator.hpp"
using DoubleVector = dealii::LinearAlgebra::distributed::Vector<double>;

template class solver::MultigridPreconditioner<3u, float, solver::ZeroModeOperator<3u, float>>;
using MGPrecZeroFloat3 = solver::MultigridPreconditioner<3u, float, solver::ZeroModeOperator<3u, float>>;
template void MGPrecZeroFloat3::vmult<DoubleVector>(DoubleVector &dst, const DoubleVector &src) const;
template void MGPrecZeroFloat3::Tvmult<DoubleVector>(DoubleVector &dst, const DoubleVector &src) const;

template class solver::MultigridPreconditioner<2u, float, solver::ZeroModeOperator<2u, float>>;
using MGPrecZeroFloat2 = solver::MultigridPreconditioner<2u, float, solver::ZeroModeOperator<2u, float>>;
template void MGPrecZeroFloat2::vmult<DoubleVector>(DoubleVector &dst, const DoubleVector &src) const;
template void MGPrecZeroFloat2::Tvmult<DoubleVector>(DoubleVector &dst, const DoubleVector &src) const;

template class solver::MultigridPreconditioner<1u, float, solver::ZeroModeOperator<1u, float>>;
using MGPrecZeroFloat1 = solver::MultigridPreconditioner<1u, float, solver::ZeroModeOperator<1u, float>>;
template void MGPrecZeroFloat1::vmult<DoubleVector>(DoubleVector &dst, const DoubleVector &src) const;
template void MGPrecZeroFloat1::Tvmult<DoubleVector>(DoubleVector &dst, const DoubleVector &src) const;

template class solver::MultigridPreconditioner<3u, float, solver::SecondModeOperator<3u, float>>;
using MGPrecSecondFloat3 = solver::MultigridPreconditioner<3u, float, solver::SecondModeOperator<3u, float>>;
template void MGPrecSecondFloat3::vmult<DoubleVector>(DoubleVector &dst, const DoubleVector &src) const;
template void MGPrecSecondFloat3::Tvmult<DoubleVector>(DoubleVector &dst, const DoubleVector &src) const;

template class solver::MultigridPreconditioner<2u, float, solver::SecondModeOperator<2u, float>>;
using MGPrecSecondFloat2 = solver::MultigridPreconditioner<2u, float, solver::SecondModeOperator<2u, float>>;
template void MGPrecSecondFloat2::vmult<DoubleVector>(DoubleVector &dst, const DoubleVector &src) const;
template void MGPrecSecondFloat2::Tvmult<DoubleVector>(DoubleVector &dst, const DoubleVector &src) const;

template class solver::MultigridPreconditioner<1u, float, solver::SecondModeOperator<1u, float>>;
using MGPrecSecondFloat1 = solver::MultigridPreconditioner<1u, float, solver::SecondModeOperator<1u, float>>;
template void MGPrecSecondFloat1::vmult<DoubleVector>(DoubleVector &dst, const DoubleVector &src) const;
template void MGPrecSecondFloat1::Tvmult<DoubleVector>(DoubleVector &dst, const DoubleVector &src) const;


#include "SP3Operator.hpp"
#include "BlockDiagonalPreconditioner.hpp"
using DoubleBlockVector = dealii::LinearAlgebra::distributed::BlockVector<double>;
using FloatBlockVector = dealii::LinearAlgebra::distributed::BlockVector<float>;

template class solver::MultigridPreconditioner<
  1u,
  float,
  solver::SP3Operator<1u, float>,
  FloatBlockVector,
  solver::MGTransferBlockGlobalCoarsening<1u, float>,
  solver::BlockDiagonalPreconditioner<float>,
  solver::MGCoarseGridTrilinosBlockWrapper<1u, float>>;
using MGPrecSP3Float1 = solver::MultigridPreconditioner<
  1u,
  float,
  solver::SP3Operator<1u, float>,
  FloatBlockVector,
  solver::MGTransferBlockGlobalCoarsening<1u, float>,
  solver::BlockDiagonalPreconditioner<float>,
  solver::MGCoarseGridTrilinosBlockWrapper<1u, float>>;
template void MGPrecSP3Float1::vmult<DoubleBlockVector>(DoubleBlockVector &dst, const DoubleBlockVector &src) const;
template void MGPrecSP3Float1::Tvmult<DoubleBlockVector>(DoubleBlockVector &dst, const DoubleBlockVector &src) const;

template class solver::MultigridPreconditioner<
  2u,
  float,
  solver::SP3Operator<2u, float>,
  FloatBlockVector,
  solver::MGTransferBlockGlobalCoarsening<2u, float>,
  solver::BlockDiagonalPreconditioner<float>,
  solver::MGCoarseGridTrilinosBlockWrapper<2u, float>>;
using MGPrecSP3Float2 = solver::MultigridPreconditioner<
  2u,
  float,
  solver::SP3Operator<2u, float>,
  FloatBlockVector,
  solver::MGTransferBlockGlobalCoarsening<2u, float>,
  solver::BlockDiagonalPreconditioner<float>,
  solver::MGCoarseGridTrilinosBlockWrapper<2u, float>>;
template void MGPrecSP3Float2::vmult<DoubleBlockVector>(DoubleBlockVector &dst, const DoubleBlockVector &src) const;
template void MGPrecSP3Float2::Tvmult<DoubleBlockVector>(DoubleBlockVector &dst, const DoubleBlockVector &src) const;

template class solver::MultigridPreconditioner<
  3u,
  float,
  solver::SP3Operator<3u, float>,
  FloatBlockVector,
  solver::MGTransferBlockGlobalCoarsening<3u, float>,
  solver::BlockDiagonalPreconditioner<float>,
  solver::MGCoarseGridTrilinosBlockWrapper<3u, float>>;
using MGPrecSP3Float3 = solver::MultigridPreconditioner<
  3u,
  float,
  solver::SP3Operator<3u, float>,
  FloatBlockVector,
  solver::MGTransferBlockGlobalCoarsening<3u, float>,
  solver::BlockDiagonalPreconditioner<float>,
  solver::MGCoarseGridTrilinosBlockWrapper<3u, float>>;
template void MGPrecSP3Float3::vmult<DoubleBlockVector>(DoubleBlockVector &dst, const DoubleBlockVector &src) const;
template void MGPrecSP3Float3::Tvmult<DoubleBlockVector>(DoubleBlockVector &dst, const DoubleBlockVector &src) const;
