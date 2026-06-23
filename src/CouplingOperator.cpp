#include "CouplingOperator.hpp"

#include <deal.II/base/types.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/evaluation_flags.h>


namespace solver {
  using namespace dealii;

  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::clear() {
    data.reset();
  }


  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::initialize(std::shared_ptr<const MatrixFree<dim, number>> data, std::shared_ptr<const MaterialCache<number>> material_cache, const data::MaterialData &material_data) {
    clear();
    this->data = data;
    this->material_cache = material_cache;

    const unsigned int n_batches = data->n_cell_batches() + data->n_ghost_cell_batches();

    sigma_rem.resize(n_batches);

    for (unsigned int cell_batch = 0; cell_batch < n_batches; ++cell_batch) {
      VectorizedArray<number> sig_rem_batch = 0.0;

      const unsigned int n_active = data->n_active_entries_per_cell_batch(cell_batch);
      for (unsigned int v = 0; v < n_active; ++v) {
        auto cell_iterator = data->get_cell_iterator(cell_batch, v, dof_index);
        types::material_id mat_id = cell_iterator->material_id();
        sig_rem_batch[v] = number(material_data.get_sigma_rem(mat_id, energy_group));
      }

      sigma_rem[cell_batch] = sig_rem_batch;
    }
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
    void (CouplingOperator<dim, number>::*face_operation)(const MatrixFree<dim, number> &, VectorType &, const VectorType &, const std::pair<unsigned int, unsigned int> &) const = nullptr;

    data->loop(
      &CouplingOperator::apply_cell,
      face_operation,
      &CouplingOperator::apply_boundary,
      this,
      dst,
      src,
      true, // Set dst to zero
      MatrixFree<dim, number>::DataAccessOnFaces::none,
      MatrixFree<dim, number>::DataAccessOnFaces::none
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

      const VectorizedArray<number> m12 = phi_src.read_cell_data(sigma_rem) * -(2.0 / 3.0);
      for (const unsigned int q : phi_src.quadrature_point_indices())
        phi_dst.submit_value(phi_src.get_value(q) * m12, q);

      phi_dst.integrate_scatter(EvaluationFlags::values, dst);
    }
  }

  template <unsigned int dim, typename number>
  void CouplingOperator<dim, number>::apply_boundary(const MatrixFree<dim, number> &data, VectorType &dst, const VectorType &src, const std::pair<unsigned int, unsigned int> &face_range) const {
    FEFaceEval phi_inner_src(data, true, dof_index, dof_index, /* first_selected_component */ 0);
    FEFaceEval phi_inner_dst(data, true, dof_index, dof_index, /* first_selected_component */ 0);

    for (unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_inner_src.reinit(face);
      phi_inner_dst.reinit(face);
      phi_inner_src.gather_evaluate(src, EvaluationFlags::values);

      // Check b.c. (only albedo supported at now)
      const types::boundary_id boundary_id = phi_inner_src.boundary_id();
      data::GeometryData::BoundaryConditions bc = geometry_data.get_boundary_condition(boundary_id);
      AssertThrow(bc.type != data::GeometryData::BoundaryConditions::BoundaryConditionType::Dirichlet, ExcMessage("Dirichlet boundary conditions are not implemented for CouplingOperator."));

      const VectorizedArray<number> albedo_factor = (1 - bc.param) / (1 + bc.param);
      const VectorizedArray<number> m12 = - (1.0 / 8.0) * albedo_factor;

      for (const unsigned int q : phi_inner_src.quadrature_point_indices())
        phi_inner_dst.submit_value(phi_inner_src.get_value(q) * m12, q);

      phi_inner_dst.integrate_scatter(EvaluationFlags::values, dst);
    }
  }

}



template class solver::CouplingOperator<3u, double>;

template class solver::CouplingOperator<2u, double>;

template class solver::CouplingOperator<1u, double>;
