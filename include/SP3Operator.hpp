#pragma once

#include "GeometryData.hpp"
#include "CrossSectionManager.hpp"
#include "MaterialData.hpp"
#include "BlockDiagonalPreconditioner.hpp"

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <vector>
#include <memory>


namespace solver {

  template <unsigned int dim, typename number>
  class SP3Operator : public dealii::EnableObserverPointer {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using DiagonalPreconditionerType = BlockDiagonalPreconditioner<number>;

    SP3Operator(const unsigned int p_degree, const unsigned int dof_index, const unsigned int group, const data::GeometryData &geom_data, const bool use_interior_face_terms) :
      p_degree(p_degree),
      dof_index(dof_index),
      energy_group(group),
      geometry_data(geom_data),
      use_interior_face_terms(use_interior_face_terms),
      diagonal_is_up_to_date(false)
    {};

    void clear();
    void initialize(std::shared_ptr<const dealii::MatrixFree<dim, number>>, std::shared_ptr<const MaterialCache<number>>, const data::MaterialData &, const dealii::AffineConstraints<number> &);
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> get_vector_partitioner() const;
    std::shared_ptr<const dealii::MatrixFree<dim, number>> get_matrix_free() const;
    void initialize_dof_vector(BlockVectorType &) const;
    dealii::types::global_dof_index m() const;
    number get_penalty_factor() const;
  
    void vmult(BlockVectorType &, const BlockVectorType &) const;
    void vmult_add(BlockVectorType &, const BlockVectorType &) const;
    void Tvmult(BlockVectorType &, const BlockVectorType &) const;
    void Tvmult_add(BlockVectorType &, const BlockVectorType &) const;

    void compute_diagonal();
    std::shared_ptr<DiagonalPreconditionerType> get_matrix_diagonal_inverse() const;

    void compute_matrix(const dealii::DoFHandler<dim> &, dealii::TrilinosWrappers::SparseMatrix &, const dealii::AffineConstraints<number> &) const;
    void compute_matrix_on_active_dofs(const dealii::DoFHandler<dim> &, dealii::TrilinosWrappers::SparseMatrix &, const dealii::AffineConstraints<number> &) const;

    bool uses_interior_face_terms() const { return use_interior_face_terms; }



  private:
    using TensorType = dealii::Tensor<1, 2, dealii::VectorizedArray<number>>;
    using FEEval = dealii::FEEvaluation<dim, -1, 0, 1, number>;
    using FEFaceEval = dealii::FEFaceEvaluation<dim, -1, 0, 1, number>;

    void apply_cell(const dealii::MatrixFree<dim, number> &, BlockVectorType &, const BlockVectorType &, const std::pair<unsigned int, unsigned int> &) const;
    void apply_face(const dealii::MatrixFree<dim, number> &, BlockVectorType &, const BlockVectorType &, const std::pair<unsigned int, unsigned int> &) const;
    void apply_face_T(const dealii::MatrixFree<dim, number> &, BlockVectorType &, const BlockVectorType &, const std::pair<unsigned int, unsigned int> &) const;
    void apply_boundary(const dealii::MatrixFree<dim, number> &, BlockVectorType &, const BlockVectorType &, const std::pair<unsigned int, unsigned int> &) const;

    void integrate_cell_block(FEEval &, const unsigned int, const unsigned int) const;
    void integrate_face_block(FEFaceEval &, FEFaceEval &, const unsigned int) const;
    void integrate_boundary_block(FEFaceEval &, const unsigned int, const unsigned int) const;
    void compute_scalar_diagonal(VectorType &, const unsigned int, const unsigned int) const;
    void compute_scalar_matrix(dealii::TrilinosWrappers::SparseMatrix &, const unsigned int, const unsigned int, const dealii::AffineConstraints<number> &) const;
  
    std::shared_ptr<const dealii::MatrixFree<dim, number>> data;

    const unsigned int p_degree;
    const unsigned int dof_index;
    const unsigned int energy_group;

    const data::GeometryData &geometry_data;
    const bool use_interior_face_terms;

    dealii::AlignedVector<dealii::VectorizedArray<number>> diff_coef;
    dealii::AlignedVector<dealii::VectorizedArray<number>> sigma_rem;
    dealii::AlignedVector<dealii::VectorizedArray<number>> disc_fact;

    std::shared_ptr<const MaterialCache<number>> material_cache;
    std::shared_ptr<DiagonalPreconditionerType> inverse_diagonal;
    bool diagonal_is_up_to_date;

    dealii::IndexSet constrained_dofs;
  };


}
