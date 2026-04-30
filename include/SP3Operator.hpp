#pragma once

#include "GeometryData.hpp"
#include "CrossSectionManager.hpp"
#include "MaterialData.hpp"

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <vector>
#include <memory>


namespace solver {

  template <unsigned int dim, typename number>
  class SP3Operator : public dealii::EnableObserverPointer {
  public:
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    SP3Operator(const unsigned int p_degree, const unsigned int dof_index, const data::GeometryData &geom_data) :
      p_degree(p_degree),
      dof_index(dof_index),
      geometry_data(geom_data)
    {};

    void clear();
    void initialize(std::shared_ptr<const dealii::MatrixFree<dim, number>>, std::shared_ptr<const MaterialCache<number>>, const data::MaterialData &);
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> get_vector_partitioner() const;
    std::shared_ptr<const dealii::MatrixFree<dim, number>> get_matrix_free() const;
    void initialize_dof_vector(BlockVectorType &) const;
    dealii::types::global_dof_index m() const;
    number get_penalty_factor() const;
  
    void vmult(BlockVectorType &, const BlockVectorType &) const;
    void Tvmult(BlockVectorType &, const BlockVectorType &) const;


  private:
    using TensorType = dealii::Tensor<1, 2, dealii::VectorizedArray<number>>;
    void apply_cell(const dealii::MatrixFree<dim, number> &, BlockVectorType &, const BlockVectorType &, const std::pair<unsigned int, unsigned int> &) const;
    void apply_face(const dealii::MatrixFree<dim, number> &, BlockVectorType &, const BlockVectorType &, const std::pair<unsigned int, unsigned int> &) const;
    void apply_face_T(const dealii::MatrixFree<dim, number> &, BlockVectorType &, const BlockVectorType &, const std::pair<unsigned int, unsigned int> &) const;
    void apply_boundary(const dealii::MatrixFree<dim, number> &, BlockVectorType &, const BlockVectorType &, const std::pair<unsigned int, unsigned int> &) const;
  
    std::shared_ptr<const dealii::MatrixFree<dim, number>> data;

    const unsigned int p_degree;
    const unsigned int dof_index;

    const data::GeometryData &geometry_data;

    dealii::AlignedVector<dealii::VectorizedArray<number>> diff_coef;
    dealii::AlignedVector<dealii::VectorizedArray<number>> sigma_rem;

    std::shared_ptr<const MaterialCache<number>> material_cache;
  };


}