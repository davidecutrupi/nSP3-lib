#include "SecondModeOperator.hpp"

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>


namespace solver {
  using namespace dealii;

  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::clear() {
    data.reset();
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::initialize(std::shared_ptr<const MatrixFree<dim, number>> data, std::shared_ptr<const MaterialCache<number>> material_cache) {
    clear();
    this->data = data;
    this->material_cache = material_cache;
    diagonal_is_up_to_date = false;
  }


  template <unsigned int dim, typename number>
  std::shared_ptr<const MatrixFree<dim, number>> SecondModeOperator<dim, number>::get_matrix_free() const {
    return data;
  }
  
  
  template <unsigned int dim, typename number>
  std::shared_ptr<const Utilities::MPI::Partitioner> SecondModeOperator<dim, number>::get_vector_partitioner() const {
    return data->get_dof_info(dof_index).vector_partitioner;
  }
  
  
  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::initialize_dof_vector(VectorType &vec) const {
    data->initialize_dof_vector(vec, dof_index);
  }
  
  
  template <unsigned int dim, typename number>
  number SecondModeOperator<dim, number>::el(const unsigned int row, const unsigned int col) const {
    Assert(row == col, ExcNotImplemented());
    Assert(inverse_diagonal.get() != nullptr && inverse_diagonal->m() > 0, ExcNotInitialized());
    return number(1.0) / (*inverse_diagonal)(row, row);
  }


  template <unsigned int dim, typename number>
  types::global_dof_index SecondModeOperator<dim, number>::m() const {
    Assert(data.get() != nullptr, StandardExceptions::ExcNotInitialized());
    return data->get_vector_partitioner(dof_index)->size();
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const {
    data->loop(
      &SecondModeOperator::apply_cell,
      &SecondModeOperator::apply_face,
      &SecondModeOperator::apply_boundary,
      this,
      dst,
      src,
      true, // Set dst to zero
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients
    );
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::vmult_add(VectorType &dst, const VectorType &src) const {
    data->loop(
      &SecondModeOperator::apply_cell,
      &SecondModeOperator::apply_face,
      &SecondModeOperator::apply_boundary,
      this,
      dst,
      src,
      false, // Set dst to zero
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients
    );
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::Tvmult(VectorType &dst, const VectorType &src) const {
    vmult(dst, src);
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::Tvmult_add(VectorType &dst, const VectorType &src) const {
    vmult_add(dst, src);
  }


  template <unsigned int dim, typename number>
  number SecondModeOperator<dim, number>::get_penalty_factor() const {
    return number(1.5) * p_degree * (p_degree + 1);
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::apply_cell(const MatrixFree<dim, number> &data, VectorType &dst, const VectorType &src, const std::pair<unsigned int, unsigned int> &cell_range) const {
    FEEval phi(data, dof_index, dof_index, /* first_selected_component */ 0);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.read_dof_values(src);
      integrate_cell_physics(phi, cell);
      phi.distribute_local_to_global(dst);
    }
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::apply_face(const MatrixFree<dim, number> &data, VectorType &dst, const VectorType &src, const std::pair<unsigned int, unsigned int> &face_range) const {
    FEFaceEval phi_inner(data, true, dof_index, dof_index, /* first_selected_component */ 0);
    FEFaceEval phi_outer(data, false, dof_index, dof_index, /* first_selected_component */ 0);

    for (unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_inner.reinit(face);
      phi_inner.read_dof_values(src);
      phi_outer.reinit(face);
      phi_outer.read_dof_values(src);

      integrate_face_physics(phi_inner, phi_outer);

      phi_inner.distribute_local_to_global(dst);
      phi_outer.distribute_local_to_global(dst);
    }
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::apply_boundary(const MatrixFree<dim, number> &data, VectorType &dst, const VectorType &src, const std::pair<unsigned int, unsigned int> &face_range) const {
    FEFaceEval phi_inner(data, true, dof_index, dof_index, /* first_selected_component */ 0);

    for (unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_inner.reinit(face);
      phi_inner.read_dof_values(src);
      integrate_boundary_physics(phi_inner);
      phi_inner.distribute_local_to_global(dst);
    }
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::integrate_cell_physics(FEEval &phi) const {
    integrate_cell_physics(phi, phi.get_current_cell_index());
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::integrate_cell_physics(FEEval &phi, const unsigned int cell) const {
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    const VectorizedArray<number> d = material_cache->diffusion[cell];
    const VectorizedArray<number> srem = material_cache->sigma_rem[cell];
    const VectorizedArray<number> d22 = d * (3.0 / 7.0);
    const VectorizedArray<number> m22 = (1.0 / d) * (5.0 / 27.0) + srem * (4.0 / 9.0);

    for (const unsigned int q : phi.quadrature_point_indices()) {
      phi.submit_gradient(phi.get_gradient(q) * d22, q);
      phi.submit_value(phi.get_value(q) * m22, q);
    }

    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::integrate_face_physics(FEFaceEval &phi_inner, FEFaceEval &phi_outer) const {
    phi_inner.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    phi_outer.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    // Collect diffusion coefficients of the cells
    const VectorizedArray<number> d_in = material_cache->inner_face_diffusion_interior[phi_inner.get_current_cell_index()];
    const VectorizedArray<number> d_out = material_cache->inner_face_diffusion_exterior[phi_outer.get_current_cell_index()];
    const VectorizedArray<number> half_d22_in = (d_in * number(3.0 / 7.0)) * number(0.5);
    const VectorizedArray<number> half_d22_out = (d_out * number(3.0 / 7.0)) * number(0.5);

    // Take maximum between D+ an D-
    VectorizedArray<number> d_max = std::max(d_in, d_out);

    // Get the inverse of the face length (deal.ii places the value of interest in position dim-1)
    const VectorizedArray<number> inverse_length_normal_to_face = 0.5 * (
      std::abs((phi_inner.normal_vector(0) * phi_inner.inverse_jacobian(0))[dim - 1]) +
      std::abs((phi_outer.normal_vector(0) * phi_outer.inverse_jacobian(0))[dim - 1])
    );

    // Evaluate penalty sigma
    const VectorizedArray<number> sigma = d_max * get_penalty_factor() * inverse_length_normal_to_face * (3.0 / 7.0);

    for (const unsigned int q : phi_inner.quadrature_point_indices()) {
      const auto phi_val_in = phi_inner.get_value(q);
      const auto phi_val_out = phi_outer.get_value(q);
      const auto phi_grad_in = phi_inner.get_normal_derivative(q);
      const auto phi_grad_out = phi_outer.get_normal_derivative(q);

      // Evaluate the jump of the flux and the avg normal derivative
      const VectorizedArray<number> jump = phi_val_in - phi_val_out;
      const VectorizedArray<number> avg_normal_derivative = (half_d22_in * phi_grad_in + half_d22_out * phi_grad_out);
      
      // test_by_value includes also the penalty term
      const VectorizedArray<number> test_by_value = jump * sigma - avg_normal_derivative;

      phi_inner.submit_value(test_by_value, q);
      phi_outer.submit_value(-test_by_value, q);

      phi_inner.submit_normal_derivative(-jump * half_d22_in, q);
      phi_outer.submit_normal_derivative(-jump * half_d22_out, q);
    }

    phi_inner.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    phi_outer.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::integrate_boundary_physics(FEFaceEval &phi_inner) const {
    phi_inner.evaluate(EvaluationFlags::values);

    // Check b.c. (only albedo supported at now)
    const types::boundary_id boundary_id = phi_inner.boundary_id();
    data::GeometryData::BoundaryConditions bc = geometry_data.get_boundary_condition(boundary_id);
    Assert(bc.type != data::GeometryData::BoundaryConditions::BoundaryConditionType::Dirichlet, StandardExceptions::ExcNotImplemented());
    
    // Evaluate mass matrix using albedo parameter
    const VectorizedArray<number> albedo_factor = number((1 - bc.param) / (1 + bc.param));
    const VectorizedArray<number> m22 = (7.0 / 24.0) * albedo_factor;

    for (const unsigned int q : phi_inner.quadrature_point_indices()) {
      phi_inner.submit_value(phi_inner.get_value(q) * m22, q);
    }

    phi_inner.integrate(EvaluationFlags::values);
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::compute_matrix(TrilinosWrappers::SparseMatrix &matrix) const {
    AffineConstraints<number> dummy;
    dummy.close();
    
    MatrixFreeTools::compute_matrix(
      *data,
      dummy,
      matrix,
      &SecondModeOperator::integrate_cell_physics,
      &SecondModeOperator::integrate_face_physics,
      &SecondModeOperator::integrate_boundary_physics,
      this,
      dof_index,
      dof_index,
      0 /* first_selected_component */ 
    );
  }


  template <unsigned int dim, typename number>
  void SecondModeOperator<dim, number>::compute_diagonal() {
    if (diagonal_is_up_to_date)
      return;

    if (!inverse_diagonal) {
      inverse_diagonal = std::make_shared<DiagonalMatrix<VectorType>>();
      VectorType &inverse_diagonal_vector = inverse_diagonal->get_vector();
      data->initialize_dof_vector(inverse_diagonal_vector, dof_index);
    }

    VectorType &inverse_diagonal_vector = inverse_diagonal->get_vector();
    inverse_diagonal_vector = 0.0;
    
    MatrixFreeTools::compute_diagonal(
      *data, 
      inverse_diagonal_vector,
      &SecondModeOperator::integrate_cell_physics,
      &SecondModeOperator::integrate_face_physics,
      &SecondModeOperator::integrate_boundary_physics,
      this,
      dof_index,
      dof_index,
      0 /* first_selected_component */ 
    );
    
    for (unsigned int i = 0; i < inverse_diagonal_vector.locally_owned_size(); ++i) {
      const number diag_val = inverse_diagonal_vector.local_element(i);
      Assert(diag_val > 0.0, ExcMessage("Zero or negative diagonal entry detected!"));
      if (std::abs(diag_val) > 1e-12) 
        inverse_diagonal_vector.local_element(i) = number(1.0) / diag_val;
    }
    
    // Update ghost values
    inverse_diagonal_vector.update_ghost_values();

    // Flag diagonal as updated
    diagonal_is_up_to_date = true;
  }


  template <unsigned int dim, typename number>
  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>> SecondModeOperator<dim, number>::get_matrix_diagonal_inverse() const {
    return inverse_diagonal;
  }

}



template class solver::SecondModeOperator<3u, double>;
template class solver::SecondModeOperator<2u, double>;
template class solver::SecondModeOperator<1u, double>;

template class solver::SecondModeOperator<3u, float>;
template class solver::SecondModeOperator<2u, float>;
template class solver::SecondModeOperator<1u, float>;
