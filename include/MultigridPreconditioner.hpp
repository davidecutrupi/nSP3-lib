#pragma once

#include "MultigridWrappers.hpp"

#include <deal.II/base/mg_level_object.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <memory>
#include <vector>


namespace solver {

  template <
    unsigned int dim,
    typename number,
    typename OperatorType,
    typename VectorType = dealii::LinearAlgebra::distributed::Vector<number>,
    typename TransferType = dealii::MGTransferMatrixFree<dim, number>,
    typename SmootherPreconditionerType = dealii::DiagonalMatrix<VectorType>,
    typename CoarseWrapperType = MGCoarseGridTrilinosWrapper<dim, VectorType>>
  class MultigridPreconditioner {
  public:
    using SmootherType = dealii::PreconditionChebyshev<OperatorType, VectorType, SmootherPreconditionerType>;

    MultigridPreconditioner() = default;

    void clear();
    void initialize(const dealii::DoFHandler<dim> &, const std::vector<std::shared_ptr<OperatorType>> &, std::shared_ptr<TransferType>);
    template <typename OtherVectorType> void vmult(OtherVectorType &, const OtherVectorType &) const;
    template <typename OtherVectorType> void Tvmult(OtherVectorType &, const OtherVectorType &) const;


  private:
    std::shared_ptr<TransferType> transfer;
  
    std::unique_ptr<LevelMatrixWrapper<VectorType, OperatorType>> mg_matrix_wrapper;
    
    std::unique_ptr<dealii::MGLevelObject<SmootherType>> level_smoothers;
    std::unique_ptr<LevelSmootherWrapper<VectorType, SmootherType>> mg_smoother_wrapper;

    std::unique_ptr<dealii::TrilinosWrappers::SparseMatrix> coarse_matrix;
    std::unique_ptr<CoarseWrapperType> mg_coarse_wrapper;

    std::unique_ptr<dealii::Multigrid<VectorType>> multigrid;
    std::unique_ptr<dealii::PreconditionMG<dim, VectorType, TransferType>> preconditioner;

  };
}
