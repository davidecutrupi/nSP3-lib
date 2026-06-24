#pragma once

#include "SolverParameters.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <memory>
#include <stdexcept>


namespace solver::fe_discretization {

  inline bool uses_interior_face_terms(const SolverParameters &parameters) {
    return parameters.fe_type == "DG";
  }


  template <unsigned int dim>
  std::unique_ptr<dealii::FiniteElement<dim>> make_scalar_fe(const SolverParameters &parameters, const unsigned int degree) {
    if (parameters.fe_type == "DG")
      return std::make_unique<dealii::FE_DGQ<dim>>(degree);
    else if (parameters.fe_type == "CG")
      return std::make_unique<dealii::FE_Q<dim>>(degree);
    throw std::runtime_error("Unknown FE type");
  }


  template <unsigned int dim, typename number>
  void build_constraints(const dealii::DoFHandler<dim> &dof_handler, dealii::AffineConstraints<number> &constraints, const SolverParameters &parameters) {
    constraints.clear();

    const dealii::IndexSet locally_owned = dof_handler.locally_owned_dofs();
    const dealii::IndexSet locally_relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    constraints.reinit(locally_owned, locally_relevant_dofs);

    if (parameters.fe_type == "CG")
      dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    constraints.close();
  }


  template <unsigned int dim, typename number>
  typename dealii::MatrixFree<dim, number>::AdditionalData make_matrix_free_additional_data(const bool use_interior_face_terms) {
    typename dealii::MatrixFree<dim, number>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = dealii::MatrixFree<dim, number>::AdditionalData::none;
    additional_data.mapping_update_flags = (dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    if (use_interior_face_terms)
      additional_data.mapping_update_flags_inner_faces = (dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_normal_vectors | dealii::update_inverse_jacobians);

    additional_data.mapping_update_flags_boundary_faces = (dealii::update_values | dealii::update_JxW_values);

    return additional_data;
  }

}
