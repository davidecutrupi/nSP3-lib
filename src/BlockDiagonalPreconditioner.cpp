#include "BlockDiagonalPreconditioner.hpp"

#include <deal.II/base/exceptions.h>


namespace solver {
  using namespace dealii;

  template <typename number>
  void BlockDiagonalPreconditioner<number>::clear() {
    inv00.reinit(0);
    inv02.reinit(0);
    inv22.reinit(0);
  }


  template <typename number>
  void BlockDiagonalPreconditioner<number>::initialize(const VectorType &diag00, const VectorType &diag02, const VectorType &diag22) {
    inv00.reinit(diag00, true);
    inv02.reinit(diag00, true);
    inv22.reinit(diag00, true);

    // Compute the inverse of the 2x2 local matrix
    for (unsigned int i = 0; i < diag00.locally_owned_size(); ++i) {
      const number a00 = diag00.local_element(i);
      const number a02 = diag02.local_element(i);
      const number a22 = diag22.local_element(i);
      const number determinant = a00 * a22 - a02 * a02;

      AssertThrow(a00 > number(0.0), ExcMessage("Non-positive SP3 zero-mode diagonal in block diagonal setup."));
      AssertThrow(a22 > number(0.0), ExcMessage("Non-positive SP3 second-mode diagonal in block diagonal setup."));
      AssertThrow(determinant > number(0.0), ExcMessage("Singular or indefinite local SP3 2x2 diagonal block."));

      inv00.local_element(i) = a22 / determinant;
      inv02.local_element(i) = -a02 / determinant;
      inv22.local_element(i) = a00 / determinant;
    }

    inv00.update_ghost_values();
    inv02.update_ghost_values();
    inv22.update_ghost_values();
  }


  template <typename number>
  types::global_dof_index BlockDiagonalPreconditioner<number>::m() const {
    return 2 * inv00.size();
  }


  template <typename number>
  types::global_dof_index BlockDiagonalPreconditioner<number>::n() const {
    return m();
  }

  
  template <typename number>
  void BlockDiagonalPreconditioner<number>::vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    for (unsigned int i = 0; i < inv00.locally_owned_size(); ++i) {
      const number src0 = src.block(0).local_element(i);
      const number src2 = src.block(1).local_element(i);

      dst.block(0).local_element(i) = inv00.local_element(i) * src0 + inv02.local_element(i) * src2;
      dst.block(1).local_element(i) = inv02.local_element(i) * src0 + inv22.local_element(i) * src2;
    }
  }


  template <typename number>
  void BlockDiagonalPreconditioner<number>::Tvmult(BlockVectorType &dst, const BlockVectorType &src) const {
    vmult(dst, src);
  }
  
}


template class solver::BlockDiagonalPreconditioner<float>;
template class solver::BlockDiagonalPreconditioner<double>;
