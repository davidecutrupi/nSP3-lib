#pragma once

#include <deal.II/base/index_set.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <memory>

namespace solver {

  template <typename number>
  class BlockDiagonalPreconditioner {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    void initialize(const VectorType &, const VectorType &, const VectorType &, const dealii::IndexSet &);
    void clear();

    dealii::types::global_dof_index m() const;
    dealii::types::global_dof_index n() const;

    void vmult(BlockVectorType &, const BlockVectorType &) const;
    void Tvmult(BlockVectorType &, const BlockVectorType &) const;

  private:
    VectorType inv00;
    VectorType inv02;
    VectorType inv22;
  };

}
