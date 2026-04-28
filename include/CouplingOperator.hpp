#pragma once

#include "GeometryData.hpp"
#include "CrossSectionManager.hpp"

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <vector>
#include <memory>


namespace solver {

  template <unsigned int dim, typename number>
  class CouplingOperator : public dealii::EnableObserverPointer {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    CouplingOperator(const unsigned int dof_index, const data::GeometryData &geom_data) :
      dof_index(dof_index),
      geometry_data(geom_data)
    {};

    void clear();
    void initialize(std::shared_ptr<const dealii::MatrixFree<dim, number>>, std::shared_ptr<const MaterialCache<number>>);
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> get_vector_partitioner() const;
    std::shared_ptr<const dealii::MatrixFree<dim, number>> get_matrix_free() const;
    void initialize_dof_vector(VectorType &) const;
    dealii::types::global_dof_index m() const;
  
    void vmult(VectorType &, const VectorType &) const;
    void Tvmult(VectorType &, const VectorType &) const;
  

  private:
    void apply_cell(const dealii::MatrixFree<dim, number> &, VectorType &, const VectorType &, const std::pair<unsigned int, unsigned int> &) const;
    void apply_cell_T(const dealii::MatrixFree<dim, number> &, VectorType &, const VectorType &, const std::pair<unsigned int, unsigned int> &) const;
    void apply_face(const dealii::MatrixFree<dim, number> &, VectorType &, const VectorType &, const std::pair<unsigned int, unsigned int> &) const;
    void apply_boundary(const dealii::MatrixFree<dim, number> &, VectorType &, const VectorType &, const std::pair<unsigned int, unsigned int> &) const;
    void apply_boundary_T(const dealii::MatrixFree<dim, number> &, VectorType &, const VectorType &, const std::pair<unsigned int, unsigned int> &) const;
  
    using FEEval = dealii::FEEvaluation<dim, -1, 0, 1, number>;
    using FEFaceEval = dealii::FEFaceEvaluation<dim, -1, 0, 1, number>;
    void integrate_cell_physics(FEEval &, FEEval &, const unsigned int) const;
    void integrate_boundary_physics(FEFaceEval &, FEFaceEval &) const;

    std::shared_ptr<const dealii::MatrixFree<dim, number>> data;

    const unsigned int dof_index;
    const data::GeometryData &geometry_data;

    std::shared_ptr<const MaterialCache<number>> material_cache;
  };


}