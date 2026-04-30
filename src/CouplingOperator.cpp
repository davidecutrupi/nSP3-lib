#include "CouplingOperator.hpp"

#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/evaluation_flags.h>


namespace solver {
  using namespace dealii;

  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::clear() {
    data.reset();
  }


  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::initialize(std::shared_ptr<const MatrixFree<dim, number>> data, std::shared_ptr<const MaterialCache<number>> material_cache) {
    clear();
    this->data = data;
    this->material_cache = material_cache;
  }


  template <unsigned int dim, typename number>
  std::shared_ptr<const MatrixFree<dim, number>> CouplingOperator<dim, number>::get_matrix_free() const {
    return data;
  }
  
  
  template <unsigned int dim, typename number>
  std::shared_ptr<const Utilities::MPI::Partitioner> CouplingOperator<dim, number>::get_vector_partitioner() const {
    return data->get_dof_info(dof_index).vector_partitioner;
  }
  
  
  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::initialize_dof_vector(VectorType &vec) const {
    data->initialize_dof_vector(vec, dof_index);
  }
  

  template <unsigned int dim, typename number>
  types::global_dof_index CouplingOperator<dim, number>::m() const {
    Assert(data.get() != nullptr, StandardExceptions::ExcNotInitialized());
    return data->get_vector_partitioner(dof_index)->size();
  }


  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const {
    data->loop(
      &CouplingOperator::apply_cell,
      &CouplingOperator::apply_face,
      &CouplingOperator::apply_boundary,
      this,
      dst,
      src,
      true, // Set dst to zero
      MatrixFree<dim, number>::DataAccessOnFaces::values,
      MatrixFree<dim, number>::DataAccessOnFaces::values
    );
  }


  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::Tvmult(VectorType &dst, const VectorType &src) const {
    vmult(dst, src);
  }


  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::apply_cell(const MatrixFree<dim, number> &data, VectorType &dst, const VectorType &src, const std::pair<unsigned int, unsigned int> &cell_range) const {
    FEEval phi_src(data, dof_index, dof_index, /* first_selected_component */ 0);
    FEEval phi_dst(data, dof_index, dof_index, /* first_selected_component */ 0);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_src.reinit(cell);
      phi_dst.reinit(cell);
      phi_src.gather_evaluate(src, EvaluationFlags::values);
      integrate_cell_physics(phi_src, phi_dst, cell);
      phi_dst.integrate_scatter(EvaluationFlags::values, dst);
    }
  }

  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::apply_face(const MatrixFree<dim, number> &, VectorType &, const VectorType &, const std::pair<unsigned int, unsigned int> &) const {
    // No face contributions
  }


  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::apply_boundary(const MatrixFree<dim, number> &data, VectorType &dst, const VectorType &src, const std::pair<unsigned int, unsigned int> &face_range) const {
    FEFaceEval phi_inner_src(data, true, dof_index, dof_index, /* first_selected_component */ 0);
    FEFaceEval phi_inner_dst(data, true, dof_index, dof_index, /* first_selected_component */ 0);

    for (unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_inner_src.reinit(face);
      phi_inner_dst.reinit(face);
      phi_inner_src.gather_evaluate(src, EvaluationFlags::values);
      integrate_boundary_physics(phi_inner_src, phi_inner_dst);
      phi_inner_dst.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::integrate_cell_physics(FEEval &phi_src, FEEval &phi_dst, const unsigned int cell) const {
    const VectorizedArray<number> m12 = material_cache->sigma_rem[cell] * -(2.0 / 3.0);
    for (const unsigned int q : phi_src.quadrature_point_indices()) {
      phi_dst.submit_value(phi_src.get_value(q) * m12, q);
    }
  }


  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::integrate_boundary_physics(FEFaceEval &phi_src, FEFaceEval &phi_dst) const {
    // Check b.c. (only albedo supported at now)
    const types::boundary_id boundary_id = phi_src.boundary_id();
    data::GeometryData::BoundaryConditions bc = geometry_data.get_boundary_condition(boundary_id);
    Assert(bc.type != data::GeometryData::BoundaryConditions::BoundaryConditionType::Dirichlet, StandardExceptions::ExcNotImplemented());
    
    const VectorizedArray<number> albedo_factor = (1 - bc.param) / (1 + bc.param);
    const VectorizedArray<number> m12 = - (1.0 / 8.0) * albedo_factor;

    for (const unsigned int q : phi_src.quadrature_point_indices()) {
      phi_dst.submit_value(phi_src.get_value(q) * m12, q);
    }
  }

}



template class solver::CouplingOperator<3u, double>;

template class solver::CouplingOperator<2u, double>;