#include "SP3Operator.hpp"

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/base/types.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/multigrid/mg_tools.h>

#include <limits>


namespace solver {
  using namespace dealii;
  
  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::clear() {
    data.reset();
    inverse_diagonal.reset();
    diagonal_is_up_to_date = false;
  }
  
  
  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::initialize(std::shared_ptr<const MatrixFree<dim, number>> data, std::shared_ptr<const MaterialCache<number>> material_cache, const data::MaterialData &material_data) {
    clear();
    this->data = data;
    this->material_cache = material_cache;
    diagonal_is_up_to_date = false;

    const unsigned int n_batches = data->n_cell_batches() + data->n_ghost_cell_batches();
    
    diff_coef.resize(n_batches);
    sigma_rem.resize(n_batches);

    for (unsigned int cell_batch = 0; cell_batch < n_batches; ++cell_batch) {
      VectorizedArray<number> diff_batch = 0.0;
      VectorizedArray<number> sig_rem_batch = 0.0;
      
      const unsigned int n_active = data->n_active_entries_per_cell_batch(cell_batch);
      for (unsigned int v = 0; v < n_active; ++v) {
        auto cell_iterator = data->get_cell_iterator(cell_batch, v, dof_index);
        types::material_id mat_id = cell_iterator->material_id();
        diff_batch[v] = number(material_data.get_diffusion(mat_id, dof_index));
        sig_rem_batch[v] = number(material_data.get_sigma_rem(mat_id, dof_index));
      }

      diff_coef[cell_batch] = diff_batch;
      sigma_rem[cell_batch] = sig_rem_batch;
    }
  }


  template <unsigned int dim, typename number>
  std::shared_ptr<const MatrixFree<dim, number>> SP3Operator<dim, number>::get_matrix_free() const {
    return data;
  }
  
  
  template <unsigned int dim, typename number>
  std::shared_ptr<const Utilities::MPI::Partitioner> SP3Operator<dim, number>::get_vector_partitioner() const {
    return data->get_dof_info(dof_index).vector_partitioner;
  }
  
  
  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::initialize_dof_vector(BlockVectorType &vec) const {
    Assert(vec.n_blocks() == 2, ExcMessage("SP3Operator expects exactly two blocks for SP3 mode 0 and mode 2."));

    data->initialize_dof_vector(vec.block(0), dof_index);
    data->initialize_dof_vector(vec.block(1), dof_index);
    vec.collect_sizes();

    Assert(vec.block(0).size() == vec.block(1).size(), ExcMessage("SP3 mode blocks must have the same scalar DoF-space size."));
    Assert(vec.block(0).locally_owned_elements() == vec.block(1).locally_owned_elements(), ExcMessage("SP3 mode blocks must use identical locally owned scalar DoF partitions."));
  }
  

  template <unsigned int dim, typename number>
  types::global_dof_index SP3Operator<dim, number>::m() const {
    Assert(data.get() != nullptr, StandardExceptions::ExcNotInitialized());
    return 2 * data->get_vector_partitioner(dof_index)->size();
  }


  template <unsigned int dim, typename number>
  number SP3Operator<dim, number>::get_penalty_factor() const {
    return number(2.0) * p_degree * (p_degree + 1);
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    data->loop(
      &SP3Operator::apply_cell,
      &SP3Operator::apply_face,
      &SP3Operator::apply_boundary,
      this,
      dst,
      src,
      true, // Set dst to zero
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients
    );
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::vmult_add(BlockVectorType &dst, const BlockVectorType &src) const {
    data->loop(
      &SP3Operator::apply_cell,
      &SP3Operator::apply_face,
      &SP3Operator::apply_boundary,
      this,
      dst,
      src,
      false, // Do not set dst to zero
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients
    );
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::Tvmult(BlockVectorType &dst, const BlockVectorType &src) const {
    data->loop(
      &SP3Operator::apply_cell,
      &SP3Operator::apply_face_T,
      &SP3Operator::apply_boundary,
      this,
      dst,
      src,
      true, // Set dst to zero
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients
    );
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::Tvmult_add(BlockVectorType &dst, const BlockVectorType &src) const {
    data->loop(
      &SP3Operator::apply_cell,
      &SP3Operator::apply_face_T,
      &SP3Operator::apply_boundary,
      this,
      dst,
      src,
      false, // Do not set dst to zero
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients
    );
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::apply_cell(const MatrixFree<dim, number> &data, BlockVectorType &dst, const BlockVectorType &src, const std::pair<unsigned int, unsigned int> &cell_range) const {
    FEEvaluation<dim, -1, 0, 1, number> phi0(data, dof_index, dof_index);
    FEEvaluation<dim, -1, 0, 1, number> phi2(data, dof_index, dof_index);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi0.reinit(cell);
      phi2.reinit(cell);
      phi0.gather_evaluate(src.block(0), EvaluationFlags::gradients | EvaluationFlags::values);
      phi2.gather_evaluate(src.block(1), EvaluationFlags::gradients | EvaluationFlags::values);
      
      const VectorizedArray<number> d = phi0.read_cell_data(diff_coef);
      const VectorizedArray<number> srem = phi0.read_cell_data(sigma_rem);

      const VectorizedArray<number> d22 = d * (3.0 / 7.0);
      const VectorizedArray<number> m12 = srem * -(2.0 / 3.0);
      const VectorizedArray<number> m22 = (1.0 / d) * (5.0 / 27.0) + srem * (4.0 / 9.0);

      for (const unsigned int q : phi0.quadrature_point_indices()) {
        const VectorizedArray<number> phi0_val = phi0.get_value(q);
        const VectorizedArray<number> phi2_val = phi2.get_value(q);

        Tensor<1, 2, Tensor<1, dim, VectorizedArray<number>>> grad_block;
        grad_block[0] = phi0.get_gradient(q) * d;
        grad_block[1] = phi2.get_gradient(q) * d22;

        Tensor<1, 2, VectorizedArray<number>> val_block;
        val_block[0] = phi0_val * srem + phi2_val * m12;
        val_block[1] = phi2_val * m22 + phi0_val * m12;

        phi0.submit_gradient(grad_block[0], q);
        phi0.submit_value(val_block[0], q);
        phi2.submit_gradient(grad_block[1], q);
        phi2.submit_value(val_block[1], q);
      }
      phi0.integrate_scatter(EvaluationFlags::gradients | EvaluationFlags::values, dst.block(0));
      phi2.integrate_scatter(EvaluationFlags::gradients | EvaluationFlags::values, dst.block(1));
    }
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::apply_face(const MatrixFree<dim, number> &data, BlockVectorType &dst, const BlockVectorType &src, const std::pair<unsigned int, unsigned int> &face_range) const {
    FEFaceEvaluation<dim, -1, 0, 1, number> phi0_inner(data, true, dof_index, dof_index);
    FEFaceEvaluation<dim, -1, 0, 1, number> phi2_inner(data, true, dof_index, dof_index);
    FEFaceEvaluation<dim, -1, 0, 1, number> phi0_outer(data, false, dof_index, dof_index);
    FEFaceEvaluation<dim, -1, 0, 1, number> phi2_outer(data, false, dof_index, dof_index);

    for (unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi0_inner.reinit(face);
      phi2_inner.reinit(face);
      phi0_outer.reinit(face);
      phi2_outer.reinit(face);

      phi0_inner.gather_evaluate(src.block(0), EvaluationFlags::values | EvaluationFlags::gradients);
      phi2_inner.gather_evaluate(src.block(1), EvaluationFlags::values | EvaluationFlags::gradients);
      phi0_outer.gather_evaluate(src.block(0), EvaluationFlags::values | EvaluationFlags::gradients);
      phi2_outer.gather_evaluate(src.block(1), EvaluationFlags::values | EvaluationFlags::gradients);

      // Collect discontinuity factors of the cellls
      const VectorizedArray<number> disc_fact_in = material_cache->inner_face_disc_fact_interior[phi0_inner.get_current_cell_index()];
      const VectorizedArray<number> disc_fact_out = material_cache->inner_face_disc_fact_exterior[phi0_outer.get_current_cell_index()];

      // Collect diffusion coefficients of the cells
      const VectorizedArray<number> half_d_in = phi0_inner.read_cell_data(diff_coef) * number(0.5);
      const VectorizedArray<number> half_d_out = phi0_outer.read_cell_data(diff_coef) * number(0.5);
      // const VectorizedArray<number> half_d22_in = half_d_in * (3.0 / 7.0);
      // const VectorizedArray<number> half_d22_out = half_d_out * (3.0 / 7.0);

      // Take maximum between D+ an D-
      VectorizedArray<number> d_max = std::max(half_d_in, half_d_out) * number(2.0);
      (void) d_max;
      const number epsilon = std::numeric_limits<number>::epsilon();
      VectorizedArray<number> d_harmonic = (number(4.0) * half_d_in * half_d_out) / (half_d_in + half_d_out + epsilon);
      (void) d_harmonic;

      // Get the inverse of the face length (deal.ii places the value of interest in position dim-1)
      const VectorizedArray<number> inverse_length_normal_to_face = 0.5 * (
        std::abs((phi0_inner.normal_vector(0) * phi0_inner.inverse_jacobian(0))[dim - 1]) +
        std::abs((phi0_outer.normal_vector(0) * phi0_outer.inverse_jacobian(0))[dim - 1])
      );

      // Evaluate penalty sigma
      const Tensor<1, 2, VectorizedArray<number>> sigma({
        d_max * get_penalty_factor() * inverse_length_normal_to_face,
        d_max * get_penalty_factor() * inverse_length_normal_to_face * (3.0 / 7.0)
      });

      for (const unsigned int q : phi0_inner.quadrature_point_indices()) {
        // Evaluate the jump of the flux for both equations (the diffusion has discontinuity factors)
        const Tensor<1, 2, VectorizedArray<number>> jump ({
          disc_fact_in * phi0_inner.get_value(q) - disc_fact_out * phi0_outer.get_value(q),
          phi2_inner.get_value(q) - phi2_outer.get_value(q)
        });

        // Avg normal derivative for both equations
        const Tensor<1, 2, VectorizedArray<number>> avg_normal_derivative ({
          half_d_in * phi0_inner.get_normal_derivative(q) + half_d_out * phi0_outer.get_normal_derivative(q),
          half_d_in * (3.0 / 7.0) * phi2_inner.get_normal_derivative(q) + half_d_out * (3.0 / 7.0) * phi2_outer.get_normal_derivative(q)
        });
        
        // test_by_value includes also the penalty term
        Tensor<1, 2, VectorizedArray<number>> test_by_value ({
          jump[0] * sigma[0] - avg_normal_derivative[0],
          jump[1] * sigma[1] - avg_normal_derivative[1]
        });

        phi0_inner.submit_value(test_by_value[0], q);
        phi2_inner.submit_value(test_by_value[1], q);

        phi0_outer.submit_value(-test_by_value[0], q);
        phi2_outer.submit_value(-test_by_value[1], q);

        phi0_inner.submit_normal_derivative(-jump[0] * half_d_in, q);
        phi2_inner.submit_normal_derivative(-jump[1] * half_d_in * (3.0 / 7.0), q);
        
        phi0_outer.submit_normal_derivative(-jump[0] * half_d_out, q);
        phi2_outer.submit_normal_derivative(-jump[1] * half_d_out * (3.0 / 7.0), q);
      }

      phi0_inner.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst.block(0));
      phi2_inner.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst.block(1));
      phi0_outer.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst.block(0));
      phi2_outer.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst.block(1));
    }
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::apply_face_T(const MatrixFree<dim, number> &data, BlockVectorType &dst, const BlockVectorType &src, const std::pair<unsigned int, unsigned int> &face_range) const {
    FEFaceEvaluation<dim, -1, 0, 1, number> phi0_inner(data, true, dof_index, dof_index);
    FEFaceEvaluation<dim, -1, 0, 1, number> phi2_inner(data, true, dof_index, dof_index);
    FEFaceEvaluation<dim, -1, 0, 1, number> phi0_outer(data, false, dof_index, dof_index);
    FEFaceEvaluation<dim, -1, 0, 1, number> phi2_outer(data, false, dof_index, dof_index);

    for (unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi0_inner.reinit(face);
      phi2_inner.reinit(face);
      phi0_outer.reinit(face);
      phi2_outer.reinit(face);

      phi0_inner.gather_evaluate(src.block(0), EvaluationFlags::values | EvaluationFlags::gradients);
      phi2_inner.gather_evaluate(src.block(1), EvaluationFlags::values | EvaluationFlags::gradients);
      phi0_outer.gather_evaluate(src.block(0), EvaluationFlags::values | EvaluationFlags::gradients);
      phi2_outer.gather_evaluate(src.block(1), EvaluationFlags::values | EvaluationFlags::gradients);

      // Collect discontinuity factors of the cellls
      const VectorizedArray<number> disc_fact_in = material_cache->inner_face_disc_fact_interior[phi0_inner.get_current_cell_index()];
      const VectorizedArray<number> disc_fact_out = material_cache->inner_face_disc_fact_exterior[phi0_outer.get_current_cell_index()];

      // Collect diffusion coefficients of the cells
      const VectorizedArray<number> half_d_in = material_cache->inner_face_diffusion_interior[phi0_inner.get_current_cell_index()] * number(0.5);
      const VectorizedArray<number> half_d_out = material_cache->inner_face_diffusion_exterior[phi0_outer.get_current_cell_index()] * number(0.5);
      const VectorizedArray<number> half_d22_in = half_d_in * (3.0 / 7.0);
      const VectorizedArray<number> half_d22_out = half_d_out * (3.0 / 7.0);

      // Take maximum between D+ an D-
      VectorizedArray<number> d_max = std::max(half_d_in, half_d_out) * number(2.0);

      // Get the inverse of the face length (deal.ii places the value of interest in position dim-1)
      const VectorizedArray<number> inverse_length_normal_to_face = 0.5 * (
        std::abs((phi0_inner.normal_vector(0) * phi0_inner.inverse_jacobian(0))[dim - 1]) +
        std::abs((phi0_outer.normal_vector(0) * phi0_outer.inverse_jacobian(0))[dim - 1])
      );

      // Evaluate penalty sigma
      const Tensor<1, 2, VectorizedArray<number>> sigma ({
        d_max * get_penalty_factor() * inverse_length_normal_to_face,
        d_max * get_penalty_factor() * inverse_length_normal_to_face * (3.0 / 7.0)
      });

      for (const unsigned int q : phi0_inner.quadrature_point_indices()) {
        // Evaluate the jump of the flux for both equations (the diffusion has discontinuity factors)
        const Tensor<1, 2, VectorizedArray<number>> jump ({
          phi0_inner.get_value(q) - phi0_outer.get_value(q),
          phi2_inner.get_value(q) - phi2_outer.get_value(q)
        });

        // Avg normal derivative for both equations
        const Tensor<1, 2, VectorizedArray<number>> avg_normal_derivative ({
          (half_d_in * phi0_inner.get_normal_derivative(q) + half_d_out * phi0_outer.get_normal_derivative(q)),
          (half_d22_in * phi2_inner.get_normal_derivative(q) + half_d22_out * phi2_outer.get_normal_derivative(q))
        });
        
        // test_by_value includes also the penalty term
        Tensor<1, 2, VectorizedArray<number>> test_by_value ({
          jump[0] * sigma[0] - avg_normal_derivative[0],
          jump[1] * sigma[1] - avg_normal_derivative[1]
        });

        phi0_inner.submit_value(disc_fact_in * test_by_value[0], q);
        phi2_inner.submit_value(test_by_value[1], q);

        phi0_outer.submit_value(-disc_fact_out * test_by_value[0], q);
        phi2_outer.submit_value(-test_by_value[1], q);

        phi0_inner.submit_normal_derivative(-jump[0] * half_d_in, q);
        phi2_inner.submit_normal_derivative(-jump[1] * half_d22_in, q);

        phi0_outer.submit_normal_derivative(-jump[0] * half_d_out, q);
        phi2_outer.submit_normal_derivative(-jump[1] * half_d22_out, q);
      }

      phi0_inner.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst.block(0));
      phi2_inner.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst.block(1));
      phi0_outer.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst.block(0));
      phi2_outer.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst.block(1));
    }
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::apply_boundary(const MatrixFree<dim, number> &data, BlockVectorType &dst, const BlockVectorType &src, const std::pair<unsigned int, unsigned int> &face_range) const {
    FEFaceEvaluation<dim, -1, 0, 1, number> phi0_inner(data, true, dof_index, dof_index);
    FEFaceEvaluation<dim, -1, 0, 1, number> phi2_inner(data, true, dof_index, dof_index);

    for (unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi0_inner.reinit(face);
      phi2_inner.reinit(face);
      phi0_inner.gather_evaluate(src.block(0), EvaluationFlags::values);
      phi2_inner.gather_evaluate(src.block(1), EvaluationFlags::values);
    
      // Check b.c. (only albedo supported at now)
      const types::boundary_id boundary_id = phi0_inner.boundary_id();
      data::GeometryData::BoundaryConditions bc = geometry_data.get_boundary_condition(boundary_id);
      AssertThrow(bc.type != data::GeometryData::BoundaryConditions::BoundaryConditionType::Dirichlet, ExcMessage("Dirichlet boundary conditions are not implemented for SP3Operator."));

      // Evaluate mass matrix using albedo parameter
      const VectorizedArray<number> albedo_factor = number((1.0 - bc.param) / (1.0 + bc.param));
      const VectorizedArray<number> m11 = (1.0 / 2.0) * albedo_factor;
      const VectorizedArray<number> m12 = - (1.0 / 8.0) * albedo_factor;
      const VectorizedArray<number> m22 = (7.0 / 24.0) * albedo_factor;

      for (const unsigned int q : phi0_inner.quadrature_point_indices()) {
        const VectorizedArray<number> phi0_val_in = phi0_inner.get_value(q);
        const VectorizedArray<number> phi2_val_in = phi2_inner.get_value(q);

        const Tensor<1, 2, VectorizedArray<number>> bc_block ({
          m11 * phi0_val_in + m12 * phi2_val_in,
          m12 * phi0_val_in + m22 * phi2_val_in
        });

        phi0_inner.submit_value(bc_block[0], q);
        phi2_inner.submit_value(bc_block[1], q);
      }

      phi0_inner.integrate_scatter(EvaluationFlags::values, dst.block(0));
      phi2_inner.integrate_scatter(EvaluationFlags::values, dst.block(1));
    }
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::integrate_cell_block(FEEval &phi, const unsigned int dst_mode, const unsigned int src_mode) const {
    const bool diagonal_block = (dst_mode == src_mode);
    const EvaluationFlags::EvaluationFlags flags = diagonal_block ? (EvaluationFlags::values | EvaluationFlags::gradients) : EvaluationFlags::values;
    phi.evaluate(flags);

    const VectorizedArray<number> d = phi.read_cell_data(diff_coef);
    const VectorizedArray<number> srem = phi.read_cell_data(sigma_rem);

    for (const unsigned int q : phi.quadrature_point_indices()) {
      if (dst_mode == 0 && src_mode == 0) {
        phi.submit_gradient(phi.get_gradient(q) * d, q);
        phi.submit_value(phi.get_value(q) * srem, q);
      }
      else if (dst_mode == 1 && src_mode == 1) {
        const VectorizedArray<number> d22 = d * number(3.0 / 7.0);
        const VectorizedArray<number> m22 = number(5.0 / 27.0) / d + srem * number(4.0 / 9.0);
        phi.submit_gradient(phi.get_gradient(q) * d22, q);
        phi.submit_value(phi.get_value(q) * m22, q);
      }
      else {
        const VectorizedArray<number> m12 = srem * number(-2.0 / 3.0);
        phi.submit_value(phi.get_value(q) * m12, q);
      }
    }

    phi.integrate(flags);
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::integrate_face_block(FEFaceEval &phi_inner, FEFaceEval &phi_outer, const unsigned int mode) const {
    phi_inner.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    phi_outer.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    const VectorizedArray<number> half_d_in = phi_inner.read_cell_data(diff_coef) * number(0.5);
    const VectorizedArray<number> half_d_out = phi_outer.read_cell_data(diff_coef) * number(0.5);
    const VectorizedArray<number> mode_scale = (mode == 0) ? number(1.0) : number(3.0 / 7.0);

    VectorizedArray<number> d_max = std::max(half_d_in, half_d_out) * number(2.0);

    const VectorizedArray<number> inverse_length_normal_to_face = 0.5 * (
      std::abs((phi_inner.normal_vector(0) * phi_inner.inverse_jacobian(0))[dim - 1]) +
      std::abs((phi_outer.normal_vector(0) * phi_outer.inverse_jacobian(0))[dim - 1])
    );

    const VectorizedArray<number> sigma = d_max * get_penalty_factor() * inverse_length_normal_to_face * mode_scale;

    const VectorizedArray<number> disc_fact_in = (mode == 0) ? material_cache->inner_face_disc_fact_interior[phi_inner.get_current_cell_index()] : VectorizedArray<number>(1.0);
    const VectorizedArray<number> disc_fact_out = (mode == 0) ? material_cache->inner_face_disc_fact_exterior[phi_outer.get_current_cell_index()] : VectorizedArray<number>(1.0);

    for (const unsigned int q : phi_inner.quadrature_point_indices()) {
      const VectorizedArray<number> jump = disc_fact_in * phi_inner.get_value(q) - disc_fact_out * phi_outer.get_value(q);
      const VectorizedArray<number> avg_normal_derivative = (half_d_in * phi_inner.get_normal_derivative(q) + half_d_out * phi_outer.get_normal_derivative(q)) * mode_scale;
      const VectorizedArray<number> test_by_value = jump * sigma - avg_normal_derivative;

      phi_inner.submit_value(test_by_value, q);
      phi_outer.submit_value(-test_by_value, q);

      phi_inner.submit_normal_derivative(-jump * half_d_in * mode_scale, q);
      phi_outer.submit_normal_derivative(-jump * half_d_out * mode_scale, q);
    }

    phi_inner.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    phi_outer.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::integrate_boundary_block(FEFaceEval &phi, const unsigned int dst_mode, const unsigned int src_mode) const {
    phi.evaluate(EvaluationFlags::values);

    const types::boundary_id boundary_id = phi.boundary_id();
    data::GeometryData::BoundaryConditions bc = geometry_data.get_boundary_condition(boundary_id);
    AssertThrow(bc.type != data::GeometryData::BoundaryConditions::BoundaryConditionType::Dirichlet, ExcMessage("Dirichlet boundary conditions are not implemented for SP3Operator."));

    const VectorizedArray<number> albedo_factor = number((1.0 - bc.param) / (1.0 + bc.param));
    VectorizedArray<number> coefficient;

    if (dst_mode == 0 && src_mode == 0)
      coefficient = number(1.0 / 2.0) * albedo_factor;
    else if (dst_mode == 1 && src_mode == 1)
      coefficient = number(7.0 / 24.0) * albedo_factor;
    else
      coefficient = number(-1.0 / 8.0) * albedo_factor;

    for (const unsigned int q : phi.quadrature_point_indices())
      phi.submit_value(phi.get_value(q) * coefficient, q);

    phi.integrate(EvaluationFlags::values);
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::compute_scalar_diagonal(VectorType &diagonal, const unsigned int dst_mode, const unsigned int src_mode) const {
    data->initialize_dof_vector(diagonal, dof_index);
    diagonal = 0.0;

    using CellOperation = std::function<void(FEEval &)>;
    using FaceOperation = std::function<void(FEFaceEval &, FEFaceEval &)>;
    using BoundaryOperation = std::function<void(FEFaceEval &)>;

    FaceOperation face_operation;
    if (dst_mode == src_mode)
      face_operation = [this, dst_mode](FEFaceEval &phi_inner, FEFaceEval &phi_outer) {
        integrate_face_block(phi_inner, phi_outer, dst_mode);
      };

    MatrixFreeTools::compute_diagonal(
      *data,
      diagonal,
      CellOperation([this, dst_mode, src_mode](FEEval &phi) { integrate_cell_block(phi, dst_mode, src_mode); }),
      face_operation,
      BoundaryOperation([this, dst_mode, src_mode](FEFaceEval &phi) { integrate_boundary_block(phi, dst_mode, src_mode); }),
      dof_index,
      dof_index,
      0 /* first_selected_component */
    );

    diagonal.update_ghost_values();
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::compute_scalar_matrix(TrilinosWrappers::SparseMatrix &matrix, const unsigned int dst_mode, const unsigned int src_mode) const {
    AffineConstraints<number> dummy;
    dummy.close();

    using CellOperation = std::function<void(FEEval &)>;
    using FaceOperation = std::function<void(FEFaceEval &, FEFaceEval &)>;
    using BoundaryOperation = std::function<void(FEFaceEval &)>;

    FaceOperation face_operation;
    if (dst_mode == src_mode)
      face_operation = [this, dst_mode](FEFaceEval &phi_inner, FEFaceEval &phi_outer) {
        integrate_face_block(phi_inner, phi_outer, dst_mode);
      };

    MatrixFreeTools::compute_matrix(
      *data,
      dummy,
      matrix,
      CellOperation([this, dst_mode, src_mode](FEEval &phi) { integrate_cell_block(phi, dst_mode, src_mode); }),
      face_operation,
      BoundaryOperation([this, dst_mode, src_mode](FEFaceEval &phi) { integrate_boundary_block(phi, dst_mode, src_mode); }),
      dof_index,
      dof_index,
      0 /* first_selected_component */
    );
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::compute_diagonal() {
    if (diagonal_is_up_to_date)
      return;

    VectorType diag00;
    VectorType diag02;
    VectorType diag22;

    compute_scalar_diagonal(diag00, 0, 0);
    compute_scalar_diagonal(diag02, 0, 1);
    compute_scalar_diagonal(diag22, 1, 1);

    inverse_diagonal = std::make_shared<DiagonalPreconditionerType>();
    inverse_diagonal->initialize(diag00, diag02, diag22);

    // Flag diagonal as updated
    diagonal_is_up_to_date = true;
  }


  template <unsigned int dim, typename number>
  std::shared_ptr<typename SP3Operator<dim, number>::DiagonalPreconditionerType> SP3Operator<dim, number>::get_matrix_diagonal_inverse() const {
    return inverse_diagonal;
  }


  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::compute_matrix(const DoFHandler<dim> &dof_handler, TrilinosWrappers::SparseMatrix &matrix) const {
    const IndexSet locally_owned_scalar_dofs = dof_handler.locally_owned_mg_dofs(0);
    const types::global_dof_index scalar_size = dof_handler.n_dofs(0);

    TrilinosWrappers::SparsityPattern flux_sparsity(locally_owned_scalar_dofs, MPI_COMM_WORLD);
    MGTools::make_flux_sparsity_pattern(dof_handler, flux_sparsity, 0);
    flux_sparsity.compress();

    TrilinosWrappers::SparsityPattern cell_sparsity(locally_owned_scalar_dofs, MPI_COMM_WORLD);
    MGTools::make_sparsity_pattern(dof_handler, cell_sparsity, 0);
    cell_sparsity.compress();

    IndexSet locally_owned_block_dofs(2 * scalar_size);
    locally_owned_block_dofs.add_indices(locally_owned_scalar_dofs);
    locally_owned_block_dofs.add_indices(locally_owned_scalar_dofs, scalar_size);
    locally_owned_block_dofs.compress();

    TrilinosWrappers::SparsityPattern block_sparsity(locally_owned_block_dofs, MPI_COMM_WORLD, flux_sparsity.max_entries_per_row() + cell_sparsity.max_entries_per_row());

    const auto add_sparsity_block = [&](
      const TrilinosWrappers::SparsityPattern &scalar_sparsity,
      const types::global_dof_index row_offset,
      const types::global_dof_index col_offset
    ) {
      std::vector<types::global_dof_index> cols;
      
      for (const auto row : locally_owned_scalar_dofs) {
        Assert(row < scalar_sparsity.n_rows(), ExcInternalError());
        Assert(scalar_sparsity.row_is_stored_locally(row), ExcMessage("Trying to read a non-locally stored Trilinos sparsity row."));

        const auto row_length = scalar_sparsity.row_length(row);

        if (row_length == static_cast<types::global_dof_index>(-1) || row_length == 0)
          continue;

        cols.clear();
        cols.reserve(row_length);

        auto entry = scalar_sparsity.begin(row);
        for (types::global_dof_index k = 0; k < row_length; ++k, ++entry)
          cols.push_back(entry->column() + col_offset);
        block_sparsity.add_entries(row + row_offset, cols.begin(), cols.end(), true);
      }
    };
   
    add_sparsity_block(flux_sparsity, 0, 0);
    add_sparsity_block(flux_sparsity, scalar_size, scalar_size);
    add_sparsity_block(cell_sparsity, 0, scalar_size);
    add_sparsity_block(cell_sparsity, scalar_size, 0);
    block_sparsity.compress();

    matrix.reinit(block_sparsity);

    const auto add_matrix_block = [&](
      const TrilinosWrappers::SparseMatrix &block,
      const types::global_dof_index row_offset,
      const types::global_dof_index col_offset
    ) {
      std::vector<types::global_dof_index> cols;
      std::vector<TrilinosScalar> vals;

      for (const auto row : locally_owned_scalar_dofs) {
        Assert(row < block.m(), ExcInternalError());
        Assert(block.in_local_range(row), ExcMessage("Trying to read a non-local Trilinos matrix row."));

        const auto row_length = block.row_length(row);
        if (row_length == 0)
          continue;

        cols.clear();
        vals.clear();
        cols.reserve(row_length);
        vals.reserve(row_length);

        auto entry = block.begin(row);
        for (unsigned int k = 0; k < row_length; ++k, ++entry) {
          const TrilinosScalar value = entry->value();

          if (value != 0.0) {
            cols.push_back(entry->column() + col_offset);
            vals.push_back(value);
          }
        }

        if (!cols.empty())
          matrix.add(row + row_offset, cols.size(), cols.data(), vals.data(), true, true);
      }
    };

    {
      TrilinosWrappers::SparseMatrix flux_block;
      flux_block.reinit(flux_sparsity);

      compute_scalar_matrix(flux_block, 0, 0);
      flux_block.compress(VectorOperation::add);
      add_matrix_block(flux_block, 0, 0);

      flux_block = 0.0;

      compute_scalar_matrix(flux_block, 1, 1);
      flux_block.compress(VectorOperation::add);
      add_matrix_block(flux_block, scalar_size, scalar_size);
    }

    {
      TrilinosWrappers::SparseMatrix cell_block;
      cell_block.reinit(cell_sparsity);
  
      compute_scalar_matrix(cell_block, 0, 1);
      cell_block.compress(VectorOperation::add);
      add_matrix_block(cell_block, 0, scalar_size);
      add_matrix_block(cell_block, scalar_size, 0);
    }

    matrix.compress(VectorOperation::add);
  }

}


template class solver::SP3Operator<3u, double>;
template class solver::SP3Operator<2u, double>;
template class solver::SP3Operator<1u, double>;

template class solver::SP3Operator<3u, float>;
template class solver::SP3Operator<2u, float>;
template class solver::SP3Operator<1u, float>;
