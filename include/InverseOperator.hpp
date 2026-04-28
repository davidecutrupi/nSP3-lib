#pragma once

#include <deal.II/lac/vector_memory.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <memory>


namespace solver {

  template <typename VectorType, typename OperatorType, typename PreconditionerType>
  class InverseOperator {
  public:
    InverseOperator(
      std::shared_ptr<const OperatorType> system_matrix,
      std::shared_ptr<const PreconditionerType> preconditioner,
      unsigned int max_iter = 100,
      double tolerance = 1e-4
    ) : 
      op_matrix(system_matrix),
      preconditioner(preconditioner),
      max_iter(max_iter),
      tolerance(tolerance),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {}

    void vmult(VectorType &, const VectorType &) const;    
    void Tvmult(VectorType &, const VectorType &) const;

    
  private:
    std::shared_ptr<const OperatorType> op_matrix;
    std::shared_ptr<const PreconditionerType> preconditioner;

    const unsigned int max_iter;
    const double tolerance;

    mutable dealii::GrowingVectorMemory<VectorType> vector_memory;

    dealii::ConditionalOStream pcout;

  };

}