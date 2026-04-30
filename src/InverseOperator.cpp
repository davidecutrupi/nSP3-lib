#include "InverseOperator.hpp"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

namespace solver {
  using namespace dealii;

  template <typename VectorType, typename OperatorType, typename PreconditionerType>
  void InverseOperator<VectorType, OperatorType, PreconditionerType>::vmult(VectorType &dst, const VectorType &src) const {
    ReductionControl reduction_control(max_iter, 1e-8, tolerance);
    try {
      SolverCG<VectorType> solver(reduction_control, vector_memory);
      solver.solve(*op_matrix, dst, src, *preconditioner);
    }
    catch (const SolverControl::NoConvergence &) { 
      pcout << "Convergence in the inverse operator not reached!!! "
            << "Iter: " << reduction_control.last_step() << "/" << max_iter
            << " | Residual: " << reduction_control.last_value() 
            << " | Target Tol: " << reduction_control.tolerance() 
            << std::endl;
      // Convergence not reached
      // Ignore it, it will be handled by the main solver
    }
  };


  template <typename VectorType, typename OperatorType, typename PreconditionerType>
  void InverseOperator<VectorType, OperatorType, PreconditionerType>::Tvmult(VectorType &dst, const VectorType &src) const {
    vmult(dst, src);
  };

}



#include "ZeroModeOperator.hpp"
#include "SecondModeOperator.hpp"
#include "MultigridPreconditioner.hpp"

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

template class solver::InverseOperator<
  dealii::LinearAlgebra::distributed::Vector<double>,
  solver::ZeroModeOperator<3u, double>,
  solver::MultigridPreconditioner<3u, float, solver::ZeroModeOperator<3u, float>>
>;

template class solver::InverseOperator<
  dealii::LinearAlgebra::distributed::Vector<double>,
  solver::ZeroModeOperator<2u, double>,
  solver::MultigridPreconditioner<2u, float, solver::ZeroModeOperator<2u, float>>
>;

template class solver::InverseOperator<
  dealii::LinearAlgebra::distributed::Vector<double>,
  solver::ZeroModeOperator<1u, double>,
  solver::MultigridPreconditioner<1u, float, solver::ZeroModeOperator<1u, float>>
>;


template class solver::InverseOperator<
  dealii::LinearAlgebra::distributed::Vector<double>,
  solver::SecondModeOperator<3u, double>,
  solver::MultigridPreconditioner<3u, float, solver::SecondModeOperator<3u, float>>
>;

template class solver::InverseOperator<
  dealii::LinearAlgebra::distributed::Vector<double>,
  solver::SecondModeOperator<2u, double>,
  solver::MultigridPreconditioner<2u, float, solver::SecondModeOperator<2u, float>>
>;

template class solver::InverseOperator<
  dealii::LinearAlgebra::distributed::Vector<double>,
  solver::SecondModeOperator<1u, double>,
  solver::MultigridPreconditioner<1u, float, solver::SecondModeOperator<1u, float>>
>;