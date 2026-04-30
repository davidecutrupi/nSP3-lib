#include "EnergyGroup.hpp"
#include "CurrentPostprocessor.hpp"
#include "GlobalTimer.hpp"

#include "BlockGSPreconditioner.hpp"
#include "CouplingOperator.hpp"

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/block_vector.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/index_set.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_coarse.h>

#include <deal.II/fe/fe_tools.h>

#include <memory>


namespace solver {
  using namespace dealii;

  template <unsigned int dim>
  void EnergyGroup<dim>::setup_coefficients(std::shared_ptr<const MatrixFree<dim, double>> mf_storage) {
    TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::Setup::Coefficients");

    const unsigned int n_groups = material_data.get_n_groups();

    material_manager = std::make_shared<CrossSectionManager<dim, double>>();
    material_manager->setup_levels(1, n_groups);
    material_manager->update(*mf_storage, 0, group, material_data);
  }
  

  template <unsigned int dim>
  void EnergyGroup<dim>::setup_feevals() {
    // Create their own FEEvaluations
    const auto mf_storage = sp3_operator->get_matrix_free();
    phi0 = std::make_unique<FEEvaluation<dim, -1, 0, 1, double>>(*mf_storage, this->group, this->group);
    phi2 = std::make_unique<FEEvaluation<dim, -1, 0, 1, double>>(*mf_storage, this->group, this->group);

    // Create other FEEvaluations
    const unsigned int n_groups = material_data.get_n_groups();
    phi_prime_old.resize(n_groups * 2);
    phi_prime.resize(n_groups * 2);

    for (unsigned int g = 0; g < n_groups; ++g) {
      phi_prime_old[g*2] = std::make_unique<FEEvaluation<dim, -1, 0, 1, double>>(*mf_storage, g, this->group);
      phi_prime[g*2] = std::make_unique<FEEvaluation<dim, -1, 0, 1, double>>(*mf_storage, g, this->group);
      phi_prime_old[g*2+1] = std::make_unique<FEEvaluation<dim, -1, 0, 1, double>>(*mf_storage, g, this->group);
      phi_prime[g*2+1] = std::make_unique<FEEvaluation<dim, -1, 0, 1, double>>(*mf_storage, g, this->group);
    }

  }


  template <unsigned int dim>
  void EnergyGroup<dim>::setup_dofs() {
    TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::Setup::Dofs");
    dof_handler.distribute_dofs(*fe);
    dof_handler.distribute_mg_dofs();
  }


  template <unsigned int dim>
  void EnergyGroup<dim>::setup_system(std::shared_ptr<const MatrixFree<dim, double>> system_mf_storage) {
    TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::Setup");
   
    setup_coefficients(system_mf_storage);

    // Setup SP3 operator
    sp3_operator = std::make_shared<SP3Operator<dim, double>>(p_degree, group, geometry_data);
    sp3_operator->initialize(system_mf_storage, material_manager->get_cache(0));

    sp3_operator->initialize_dof_vector(solution);
    sp3_operator->initialize_dof_vector(solution_old);
    sp3_operator->initialize_dof_vector(adjoint_solution);
    sp3_operator->initialize_dof_vector(adjoint_solution_old);
    sp3_operator->initialize_dof_vector(system_rhs);

    if (material_data.has_discontinuity_factors()) {
      // Setup zero mode operator
      zero_mode_operator = std::make_shared<ZeroModeOperator<dim, double>>(p_degree, group, geometry_data);
      zero_mode_operator->initialize(system_mf_storage, material_manager->get_cache(0));
  
      // Setup second mode operator
      second_mode_operator = std::make_shared<SecondModeOperator<dim, double>>(p_degree, group, geometry_data);
      second_mode_operator->initialize(system_mf_storage, material_manager->get_cache(0));
  
      // Setup coupling operator
      coupling_operator = std::make_shared<CouplingOperator<dim, double>>(group, geometry_data);
      coupling_operator->initialize(system_mf_storage, material_manager->get_cache(0));
    
      // Setup inner preconditioners
      inner_preconditioner_zero = std::make_shared<InnerPreconditionerZero>();
      inner_preconditioner_second = std::make_shared<InnerPrecontionerSecond>();
      setup_multigrid();

      // Setup inner solvers
      inner_solver_zero = std::make_shared<InnerSolverZero>(zero_mode_operator, inner_preconditioner_zero, 20, 1e-2);
      inner_solver_second = std::make_shared<InnerSolverSecond>(second_mode_operator, inner_preconditioner_second, 10, 1e-1);
    
      // Setup block gauss-seidel preconditioner
      preconditioner = std::make_shared<Preconditioner>(inner_solver_zero, inner_solver_second, coupling_operator);
      preconditioner->initialize(solution);
    }
    else {
      // Setup inner preconditioners
      inner_preconditioner_zero = std::make_shared<InnerPreconditionerZero>();
      inner_preconditioner_second = std::make_shared<InnerPrecontionerSecond>();
      setup_multigrid();

      // Setup block diagonal preconditioner
      block_diag_preconditioner = std::make_shared<BlockDiagPreconditioner>(inner_preconditioner_zero, inner_preconditioner_second);
    }
  
    setup_feevals();

    if (needs_p_transfer) { // Interpolate old solutions
      TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::HP::execute-p");

      FETools::interpolate(*transfer_old_dof_handler, h_interpolated_solution->block(0), dof_handler, solution.block(0));
      FETools::interpolate(*transfer_old_dof_handler, h_interpolated_solution->block(1), dof_handler, solution.block(1));
      
      transfer_old_dof_handler.reset();
      transfer_old_fe.reset();
      needs_p_transfer = false;

      h_interpolated_solution.reset();
    }
    else if (h_interpolated_solution) { // Only h-refinement
      solution = *h_interpolated_solution;
      h_interpolated_solution.reset();
    }
    else {
      solution = 1.0;
    }

    solution_old = solution;
  }


  template <unsigned int dim>
  void EnergyGroup<dim>::setup_multigrid() {
    TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::Setup::Multigrid");

    const unsigned int nlevels = triangulation->n_global_levels();
    
    std::vector<std::shared_ptr<ZeroModeOperator<dim, float>>> level_zero_operators(nlevels);
    std::vector<std::shared_ptr<SecondModeOperator<dim, float>>> level_second_operators(nlevels);
    std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners(nlevels);

    AffineConstraints<float> dummy;
    dummy.close();

    // Setup mg material manager
    mg_material_manager = std::make_shared<CrossSectionManager<dim, float>>();
    mg_material_manager->setup_levels(nlevels, material_data.get_n_groups());

    for (unsigned int level = 0; level < nlevels; ++level) {
      typename MatrixFree<dim, float>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
      additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values);
      additional_data.mapping_update_flags_inner_faces = (update_values | update_gradients | update_JxW_values | update_normal_vectors | update_inverse_jacobians);
      additional_data.mapping_update_flags_boundary_faces = (update_values | update_JxW_values);
      additional_data.mg_level = level;

      // TODO conviene fare un vector di storages che contiene gli storage di ogni livello della triangulation 
      // (ognuno con una lista di dof hanlder di ogni gruppo) direttamente nella classe NeutronSolver?
      // Cioè conviene fare come per lo storage di livello? 
      const auto mg_mf_storage = std::make_shared<MatrixFree<dim, float>>(); 
      mg_mf_storage->reinit(mapping, dof_handler, dummy, QGauss<1>(fe->degree + 1), additional_data);
      
      // Populate coefficients
      mg_material_manager->update(*mg_mf_storage, level, group, material_data);

      // Create the level operator and add to level_operators
      level_zero_operators[level] = std::make_shared<ZeroModeOperator<dim, float>>(p_degree, 0, geometry_data);
      level_zero_operators[level]->initialize(mg_mf_storage, mg_material_manager->get_cache(level));
      
      level_second_operators[level] = std::make_shared<SecondModeOperator<dim, float>>(p_degree, 0, geometry_data);
      level_second_operators[level]->initialize(mg_mf_storage, mg_material_manager->get_cache(level));

      partitioners[level] = mg_mf_storage->get_vector_partitioner();
    }

    std::shared_ptr<MGTransferMatrixFree<dim, float>> mg_transfer = std::make_shared<MGTransferMatrixFree<dim, float>>();
    mg_transfer->build(dof_handler, partitioners);

    inner_preconditioner_zero->initialize(dof_handler, level_zero_operators, mg_transfer);
    inner_preconditioner_second->initialize(dof_handler, level_second_operators, mg_transfer);
  }


  template <unsigned int dim> // TODO parallelize with cell_loop
  void EnergyGroup<dim>::compute_rhs(const std::vector<std::unique_ptr<EnergyGroup<dim>>> &all_groups, bool is_adjoint) {
    TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::Rhs");
    
    system_rhs = 0;

    const double c11 = 1.0;
    const double c12 = -2.0 / 3.0;
    const double c22 = 4.0 / 9.0;
        
    const auto mf_storage = sp3_operator->get_matrix_free();

    auto cache = material_manager->get_cache(0);

    // Vectors of n_q components that will be filled with all_groups contibutions
    AlignedVector<VectorizedArray<double>> term_0(phi0->n_q_points);
    AlignedVector<VectorizedArray<double>> term_1(phi0->n_q_points);

    const unsigned int n_batches = mf_storage->n_cell_batches();
    for (unsigned int cell = 0; cell < n_batches; ++cell) {
      phi0->reinit(cell);
      phi2->reinit(cell);
      
      // Reset term_0 and term_1 at each cell
      for (unsigned int q = 0; q < phi0->n_q_points; ++q) {
        term_0[q] = VectorizedArray<double>(0.0);
        term_1[q] = VectorizedArray<double>(0.0);
      }
      
      for (unsigned int g_from = 0; g_from < all_groups.size(); ++g_from) {
        // Take other cache
        auto cache_prime = all_groups[g_from]->material_manager->get_cache(0);

        // Reinit phi old and phi
        phi_prime_old[g_from*2]->reinit(cell);
        phi_prime_old[g_from*2+1]->reinit(cell);
        phi_prime[g_from*2]->reinit(cell);
        phi_prime[g_from*2+1]->reinit(cell);

        VectorizedArray<double> fission;
        VectorizedArray<double> scattering;

        if (is_adjoint) {
          fission = cache_prime->fission_distribution[cell] * cache->nu_sigma_f[cell];
          scattering = cache->sigma_s[g_from][cell];

          // Fission, take the old adjoint solution
          phi_prime_old[g_from*2]->read_dof_values(all_groups[g_from]->adjoint_solution_old.block(0));
          phi_prime_old[g_from*2]->evaluate(EvaluationFlags::values);
          phi_prime_old[g_from*2+1]->read_dof_values(all_groups[g_from]->adjoint_solution_old.block(1));
          phi_prime_old[g_from*2+1]->evaluate(EvaluationFlags::values);

          // Scattering, take the new adjoint solution of the g' group, it's considered only if g != g'
          if (this->group != g_from) {
            phi_prime[g_from*2]->read_dof_values(all_groups[g_from]->adjoint_solution.block(0));
            phi_prime[g_from*2]->evaluate(EvaluationFlags::values);
            phi_prime[g_from*2+1]->read_dof_values(all_groups[g_from]->adjoint_solution.block(1));
            phi_prime[g_from*2+1]->evaluate(EvaluationFlags::values);
          }
        }
        else {
          fission = cache->fission_distribution[cell] * cache_prime->nu_sigma_f[cell];
          scattering = cache_prime->sigma_s[this->group][cell];
          
          // Fission, take the old solution
          phi_prime_old[g_from*2]->read_dof_values(all_groups[g_from]->solution_old.block(0));
          phi_prime_old[g_from*2]->evaluate(EvaluationFlags::values);
          phi_prime_old[g_from*2+1]->read_dof_values(all_groups[g_from]->solution_old.block(1));
          phi_prime_old[g_from*2+1]->evaluate(EvaluationFlags::values);

          // Scattering, take the new solution of the g' group, it's considered only if g != g'
          if (this->group != g_from) {
            phi_prime[g_from*2]->read_dof_values(all_groups[g_from]->solution.block(0));
            phi_prime[g_from*2]->evaluate(EvaluationFlags::values);
            phi_prime[g_from*2+1]->read_dof_values(all_groups[g_from]->solution.block(1));
            phi_prime[g_from*2+1]->evaluate(EvaluationFlags::values);
          }
        }

        for (const unsigned int q : phi0->quadrature_point_indices()) {
          // Fission contribution
          term_0[q] += phi_prime_old[g_from*2]->get_value(q) * fission;
          term_1[q] += phi_prime_old[g_from*2+1]->get_value(q) * fission;

          // Add scattering contribution
          if (this->group != g_from) {
            term_0[q] += phi_prime[g_from*2]->get_value(q) * scattering;
            term_1[q] += phi_prime[g_from*2+1]->get_value(q) * scattering;
          }
        }
      }

      // Now scan again and construct the rhs
      for (const unsigned int q : phi0->quadrature_point_indices()) {
        Tensor<1, 2, VectorizedArray<double>> rhs_val;
        rhs_val[0] = c11 * term_0[q] + c12 * term_1[q];
        rhs_val[1] = c12 * term_0[q] + c22 * term_1[q];
        phi0->submit_value(rhs_val[0], q);
        phi2->submit_value(rhs_val[1], q);
      }

      phi0->integrate_scatter(EvaluationFlags::values, system_rhs.block(0));
      phi2->integrate_scatter(EvaluationFlags::values, system_rhs.block(1));
    }

    system_rhs.compress(VectorOperation::add);
  }


  template <unsigned int dim>
  void EnergyGroup<dim>::solve(bool is_adjoint) {
    TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::Solve");

    SolverControl solver_control(1000, 1e-8);
    
    try {
      if (!material_data.has_discontinuity_factors()) {
        SolverCG<BlockVectorType<double>> solver(solver_control);
        if (is_adjoint)
          solver.solve(*sp3_operator, adjoint_solution, system_rhs, *block_diag_preconditioner);
        else
          solver.solve(*sp3_operator, solution, system_rhs, *block_diag_preconditioner);
      }
      else {
        SolverFGMRES<BlockVectorType<double>> solver(solver_control);
        if (is_adjoint) {
          TransposeWrapper<SP3Operator<dim, double>, BlockVectorType<double>> tansp_op(*sp3_operator);
          TransposeWrapper<Preconditioner, BlockVectorType<double>> tansp_prec(*preconditioner);
          solver.solve(tansp_op, adjoint_solution, system_rhs, tansp_prec);
        }
        else 
          solver.solve(*sp3_operator, solution, system_rhs, *preconditioner);
      }
    } catch (const SolverControl::NoConvergence &exc) {
      std::cerr<<"Convergence not reached in solver!!"<<std::endl;
      std::cerr<<"Group: "<<group<<" | last iter: "<<solver_control.last_step();
      throw exc;
    }
  }


  template <unsigned int dim> // TODO parallelize with cell_loop
  double EnergyGroup<dim>::get_fission_source(bool is_adjoint) const {
    double local_fission_source = 0;
    
    const auto mf_storage = sp3_operator->get_matrix_free();
    auto cache = material_manager->get_cache(0);

    const unsigned int n_batches = mf_storage->n_cell_batches();
    if (is_adjoint) {
      for (unsigned int cell_batch = 0; cell_batch < n_batches; ++cell_batch) {
        phi0->reinit(cell_batch);
        phi2->reinit(cell_batch);
        
        phi0->read_dof_values(adjoint_solution.block(0));
        phi2->read_dof_values(adjoint_solution.block(1));

        phi0->evaluate(EvaluationFlags::values);
        phi2->evaluate(EvaluationFlags::values);
  
        VectorizedArray<double> cell_integral(0.0);
        for (const unsigned int q : phi0->quadrature_point_indices()) {
          VectorizedArray<double> real_flux = phi0->get_value(q) - (2.0 / 3.0) * phi2->get_value(q);
          cell_integral += real_flux * cache->fission_distribution[cell_batch] * phi0->JxW(q);
        }
  
        local_fission_source += cell_integral.sum();
      }
    }
    else {
      for (unsigned int cell_batch = 0; cell_batch < n_batches; ++cell_batch) {
        phi0->reinit(cell_batch);
        phi2->reinit(cell_batch);
        
        phi0->read_dof_values(solution.block(0));
        phi2->read_dof_values(solution.block(1));

        phi0->evaluate(EvaluationFlags::values);
        phi2->evaluate(EvaluationFlags::values);
  
        VectorizedArray<double> cell_integral(0.0);
        for (const unsigned int q : phi0->quadrature_point_indices()) {
          VectorizedArray<double> real_flux = phi0->get_value(q) - (2.0 / 3.0) * phi2->get_value(q);
          cell_integral += real_flux * cache->nu_sigma_f[cell_batch] * phi0->JxW(q);
        }
  
        local_fission_source += cell_integral.sum();
      }
    }

    return local_fission_source;
  }


  template <unsigned int dim>
  void EnergyGroup<dim>::update_solution(const double k_eff, bool is_adjoint) {
    if (is_adjoint) {
      adjoint_solution_old = adjoint_solution;
      adjoint_solution_old /= k_eff;
    }
    else {
      solution_old = solution;
      solution_old /= k_eff;
    }
  }


  template <unsigned int dim>
  void EnergyGroup<dim>::copy_forward_to_adjoint() {
    adjoint_solution = solution;
    adjoint_solution_old = solution_old;
  }


  template <unsigned int dim>
  void EnergyGroup<dim>::prepare_h_transfer() {
    TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::HP::prepare-h");
    sol_transfer = std::make_unique<SolutionTransfer<dim, VectorType<double>>>(dof_handler);
    std::vector<const VectorType<double>*> in_vectors = { &solution.block(0), &solution.block(1) };
    sol_transfer->prepare_for_coarsening_and_refinement(in_vectors);
  }


  template <unsigned int dim>
  void EnergyGroup<dim>::execute_h_transfer() {
    TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::HP::execute-h"); 

    dof_handler.distribute_dofs(*fe);

    IndexSet locally_owned = dof_handler.locally_owned_dofs();
    h_interpolated_solution = std::make_unique<BlockVectorType<double>>(2);
    h_interpolated_solution->block(0).reinit(locally_owned, MPI_COMM_WORLD);
    h_interpolated_solution->block(1).reinit(locally_owned, MPI_COMM_WORLD);

    std::vector<VectorType<double>*> out_vectors = { &h_interpolated_solution->block(0), &h_interpolated_solution->block(1) };
    sol_transfer->interpolate(out_vectors);
    sol_transfer.reset();
  }


  template <unsigned int dim>
  unsigned int EnergyGroup<dim>::get_degree() const { 
    return p_degree;
  }


  template <unsigned int dim>
  LinearAlgebra::distributed::BlockVector<double> EnergyGroup<dim>::get_solution() const { 
    return solution;
  }


  template <unsigned int dim>
  LinearAlgebra::distributed::BlockVector<double> EnergyGroup<dim>::get_adjoint_solution() const { 
    return adjoint_solution;
  }


  template <unsigned int dim>
  void EnergyGroup<dim>::set_degree(unsigned int new_degree) { 
    TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::HP::prepare-p");
    if (p_degree != new_degree) {
      transfer_old_fe = std::move(fe);
    
      transfer_old_dof_handler = std::make_unique<DoFHandler<dim>>(*triangulation);
      transfer_old_dof_handler->distribute_dofs(*transfer_old_fe);

      needs_p_transfer = true;
      p_degree = new_degree; 

      dof_handler.clear();
      fe = std::make_unique<FE_DGQ<dim>>(p_degree);
    }
  }


  template <unsigned int dim>
  void EnergyGroup<dim>::output_results(const std::string &bench_name) const {
    TimerOutput::Scope t(GlobalTimer::get(), "EnergyGroup::Output");
    
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    
    // Index sets
    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    // Get the locally relevant solution of the zero mode and the second mode
    LinearAlgebra::distributed::Vector<double> relevant_phi0, relevant_phi2, relevant_U0;
    relevant_phi0.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    relevant_U0.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    relevant_phi2.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    relevant_phi0 = solution.block(0);
    relevant_U0 = relevant_phi0;
    relevant_phi2 = solution.block(1);
    relevant_phi0.update_ghost_values();
    relevant_U0.update_ghost_values();
    relevant_phi2.update_ghost_values();

    // Compute fluxes
    relevant_phi2 /= 3.0;
    relevant_phi0.add(-2.0, relevant_phi2);

    // Add fluxes
    data_out.add_data_vector(relevant_phi0, "flux0");
    data_out.add_data_vector(relevant_phi2, "flux2");

    // Add current
    CurrentPostprocessor<dim> current_postprocessor(group, material_data);
    data_out.add_data_vector(relevant_U0, current_postprocessor);

    data_out.build_patches();
    
    const std::string filename = "../out/" + bench_name + "-" + Utilities::int_to_string(group, 2) + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
  }

}


template class solver::EnergyGroup<3u>;

template class solver::EnergyGroup<2u>;

template class solver::EnergyGroup<1u>;