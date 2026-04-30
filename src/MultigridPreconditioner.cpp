#include "MultigridPreconditioner.hpp"

#include <deal.II/base/types.h>
#include <deal.II/base/index_set.h>

#include <deal.II/multigrid/mg_tools.h>


namespace solver {
  using namespace dealii;

  template <unsigned int dim, typename number, typename OperatorType>
  void MultigridPreconditioner<dim, number, OperatorType>::clear() {
    preconditioner.reset();
    multigrid.reset();

    mg_coarse_wrapper.reset();
    coarse_matrix.reset();

    mg_smoother_wrapper.reset();
    level_smoothers.reset();

    mg_matrix_wrapper.reset();

    transfer.reset();
  }


  template <unsigned int dim, typename number, typename OperatorType>
  void MultigridPreconditioner<dim, number, OperatorType>::initialize(const DoFHandler<dim> &dof_handler, const std::vector<std::shared_ptr<OperatorType>> &level_operators, std::shared_ptr<dealii::MGTransferMatrixFree<dim, number>> mg_transfer) {
    clear();

    this->transfer = mg_transfer;
    const unsigned int nlevels = level_operators.size();

    mg_matrix_wrapper = std::make_unique<LevelMatrixWrapper<VectorType, OperatorType>>(level_operators);
    level_smoothers = std::make_unique<MGLevelObject<SmootherType>>(0, nlevels - 1);

    // Setup smoother data
    for (unsigned int level = 1; level < nlevels; ++level) {
      typename SmootherType::AdditionalData smoother_data;
      smoother_data.smoothing_range = 15.;
      smoother_data.degree = 5;
      smoother_data.eig_cg_n_iterations = 10;
    
      level_operators[level]->compute_diagonal();
      smoother_data.preconditioner = level_operators[level]->get_matrix_diagonal_inverse();

      (*level_smoothers)[level].initialize(*level_operators[level], smoother_data);
    }

    // Initialize mg_smoother
    mg_smoother_wrapper = std::make_unique<LevelSmootherWrapper<VectorType, SmootherType>>();
    mg_smoother_wrapper->initialize(level_smoothers.get());
    
    // Global index sets
    IndexSet locally_owned_level_dofs = dof_handler.locally_owned_mg_dofs(0);
    TrilinosWrappers::SparsityPattern dsp(locally_owned_level_dofs, MPI_COMM_WORLD);
    MGTools::make_flux_sparsity_pattern(dof_handler, dsp, 0);
    dsp.compress();

    // Create the coarse block sparse matrix
    coarse_matrix = std::make_unique<TrilinosWrappers::SparseMatrix>();
    coarse_matrix->reinit(dsp);

    level_operators[0]->compute_matrix(*coarse_matrix);
    coarse_matrix->compress(VectorOperation::add);

    mg_coarse_wrapper = std::make_unique<MGCoarseGridTrilinosWrapper<VectorType>>();
    mg_coarse_wrapper->initialize(*coarse_matrix);

    // In the end setup multigrid and precondtioner
    multigrid = std::make_unique<Multigrid<VectorType>>(*mg_matrix_wrapper, *mg_coarse_wrapper, *this->transfer, *mg_smoother_wrapper, *mg_smoother_wrapper);
    preconditioner = std::make_unique<PreconditionMG<dim, VectorType, MGTransferMatrixFree<dim, number>>>(dof_handler, *multigrid, *this->transfer);
  }


  template <unsigned int dim, typename number, typename OperatorType>
  template <typename OtherVectorType>
  void MultigridPreconditioner<dim, number, OperatorType>::vmult(OtherVectorType &dst, const OtherVectorType &src) const {
    preconditioner->vmult(dst, src);
  }


  template <unsigned int dim, typename number, typename OperatorType>
  template <typename OtherVectorType>
  void MultigridPreconditioner<dim, number, OperatorType>::Tvmult(OtherVectorType &dst, const OtherVectorType &src) const {
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