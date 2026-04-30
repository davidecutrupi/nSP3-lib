#include "SP3Operator.hpp"

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>
#include <iostream>


namespace solver {
  using namespace dealii;
  
  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::clear() {
    data.reset();
  }
  
  
  template <unsigned int dim, typename number>
  void SP3Operator<dim, number>::initialize(std::shared_ptr<const MatrixFree<dim, number>> data, std::shared_ptr<const MaterialCache<number>> material_cache, const data::MaterialData &material_data) {
    clear();
    this->data = data;
    this->material_cache = material_cache;

    const unsigned int n_batches = data->n_cell_batches();
    
    diff_coef.resize(n_batches);
    sigma_rem.resize(n_batches);

    for (unsigned int cell_batch = 0; cell_batch < n_batches; ++cell_batch) {
      VectorizedArray<number> diff_batch = 0.0;
      VectorizedArray<number> sig_rem_batch = 0.0;
      
      const unsigned int n_active = data->n_active_entries_per_cell_batch(cell_batch);
      for (unsigned int v = 0; v < n_active; ++v) {
        auto cell_iterator = data->get_cell_iterator(cell_batch, v, dof_index);
        types::material_id mat_id = cell_iterator->material_id();
        diff_batch[v] = material_data.get_diffusion(mat_id, dof_index);
        sig_rem_batch[v] = material_data.get_sigma_rem(mat_id, dof_index);
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
    data->initialize_dof_vector(vec.block(0), dof_index);
    data->initialize_dof_vector(vec.block(1), dof_index);
  }
  

  template <unsigned int dim, typename number>
  types::global_dof_index SP3Operator<dim, number>::m() const {
    Assert(data.get() != nullptr, StandardExceptions::ExcNotInitialized());
    return data->get_vector_partitioner(dof_index)->size();
  }


  template <unsigned int dim, typename number>
  number SP3Operator<dim, number>::get_penalty_factor() const {
    return 2.0 * p_degree * (p_degree + 1);
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
      const number epsilon = 1e-15;
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
      Assert(bc.type != data::GeometryData::BoundaryConditions::BoundaryConditionType::Dirichlet, StandardExceptions::ExcNotImplemented());

      // Evaluate mass matrix using albedo parameter
      const VectorizedArray<number> albedo_factor = (1 - bc.param) / (1 + bc.param);
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

}


template class solver::SP3Operator<3u, double>;
template class solver::SP3Operator<2u, double>;
template class solver::SP3Operator<1u, double>;