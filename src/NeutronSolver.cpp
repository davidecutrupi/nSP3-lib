#include "NeutronSolver.hpp"
#include "GlobalTimer.hpp"

#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools_topology.h>

#include <cmath>
#include <fstream>
#include <mpi.h>


namespace solver {
  using namespace dealii;

  void NeutronSolver::init_mesh() {
    TimerOutput::Scope t(GlobalTimer::get(), "Generate mesh");

    const unsigned int core_n_assemblies_x = geometry_data.get_core_n_assemblies_x();
    const unsigned int core_n_assemblies_y = geometry_data.get_core_n_assemblies_y();
    const unsigned int core_n_assemblies_z = (dim == 3) ? geometry_data.get_core_n_assemblies_z() : 1;

    const unsigned int rods_per_assembly_x = geometry_data.get_rods_per_assembly_x();
    const unsigned int rods_per_assembly_y = geometry_data.get_rods_per_assembly_y();

    const double pin_pitch_x = geometry_data.get_pin_pitch_x();
    const double pin_pitch_y = geometry_data.get_pin_pitch_y();
    const double assembly_height = geometry_data.get_assembly_height();

    std::vector<Point<dim>> vertices;
    std::vector<CellData<dim>> cells;

    std::map<std::tuple<long long, long long, long long>, unsigned int> vertex_map;
    unsigned int current_vertex_index = 0;

    // Lambda helper to create or retreive the index of a vertex
    auto get_or_create_vertex = [&](double x, double y, double z) -> unsigned int {
      auto key = std::make_tuple(static_cast<long long>(std::round(x * 1e6)), 
                                 static_cast<long long>(std::round(y * 1e6)), 
                                 static_cast<long long>(std::round(z * 1e6)));
      if (vertex_map.find(key) == vertex_map.end()) {
        vertex_map[key] = current_vertex_index++;
        if constexpr (dim == 1)
          vertices.emplace_back(x);
        else if constexpr (dim == 2)
          vertices.emplace_back(x, y);
        else 
          vertices.emplace_back(x, y, z);      
      }
      return vertex_map[key];
    };

    std::vector<Point<dim>> explicit_pin_centers;

    const unsigned int n_cells_x = core_n_assemblies_x * rods_per_assembly_x;
    const unsigned int n_cells_y = core_n_assemblies_y * rods_per_assembly_y;
    const unsigned int n_cells_z = core_n_assemblies_z;

    // Create cells
    for (unsigned int k = 0; k < n_cells_z; ++k) {
      for (unsigned int j = 0; j < n_cells_y; ++j) {
        for (unsigned int i = 0; i < n_cells_x; ++i) {
          const unsigned int ax = i / rods_per_assembly_x;
          const unsigned int cx = i % rods_per_assembly_x;
          const unsigned int ay = j / rods_per_assembly_y;
          const unsigned int cy = j % rods_per_assembly_y;
          const unsigned int az = k; 

          // Retreive material id
          const unsigned int mapped_ay = (core_n_assemblies_y - 1) - ay;
          const unsigned int mapped_cy = (rods_per_assembly_y - 1) - cy;
          int mat_id = geometry_data.get_assembly_pin(geometry_data.get_core_map(mapped_ay, ax, az), mapped_cy, cx);

          if (mat_id == -1) continue;

          const bool explicit_pins = geometry_data.get_explicit_pins_data().enabled;

          if (explicit_pins) {
            Assert(dim == 2, ExcMessage("Explicit pins currently only implemented in 2D."));
            
            const auto &ep = geometry_data.get_explicit_pins_data();
            const unsigned int n_edges = static_cast<unsigned int>(std::pow(2, 2 + ep.expected_refinements));
            const double alpha = 2 * M_PI / n_edges;
            const double effective_pin_radius = ep.radius * std::sqrt(2 * M_PI / (n_edges * std::sin(alpha)));

            const double p_x = pin_pitch_x / 2.0;
            const double p_y = pin_pitch_y / 2.0;
            const double r = effective_pin_radius / M_SQRT2;
            const double r2 = r / M_SQRT2;
            
            const double xx = i * pin_pitch_x + p_x;
            const double yy = j * pin_pitch_y + p_y;
            
            if constexpr (dim == 2)
              explicit_pin_centers.emplace_back(xx, yy);

            unsigned int v[12];
            // box
            v[0] = get_or_create_vertex(xx - p_x, yy - p_y, 0);
            v[1] = get_or_create_vertex(xx + p_x, yy - p_y, 0);
            v[2] = get_or_create_vertex(xx - p_x, yy + p_y, 0);
            v[3] = get_or_create_vertex(xx + p_x, yy + p_y, 0);
            // circle
            v[4] = get_or_create_vertex(xx - r, yy - r, 0);
            v[5] = get_or_create_vertex(xx + r, yy - r, 0);
            v[6] = get_or_create_vertex(xx - r, yy + r, 0);
            v[7] = get_or_create_vertex(xx + r, yy + r, 0);
            // circle2
            v[8] = get_or_create_vertex(xx - r2, yy - r2, 0);
            v[9] = get_or_create_vertex(xx + r2, yy - r2, 0);
            v[10] = get_or_create_vertex(xx - r2, yy + r2, 0);
            v[11] = get_or_create_vertex(xx + r2, yy + r2, 0);

            const unsigned int cell_vertices[9][4] = {
                { v[0], v[1], v[4], v[5] },
                { v[0], v[4], v[2], v[6] },
                { v[6], v[7], v[2], v[3] },
                { v[5], v[1], v[7], v[3] },
                { v[4], v[5], v[8], v[9] },
                { v[4], v[8], v[6], v[10] },
                { v[10], v[11], v[6], v[7] },
                { v[9], v[5], v[11], v[7] },
                { v[8], v[9], v[10], v[11] }
            };

            for (unsigned int c = 0; c < 9; ++c) {
              CellData<dim> cell;
              cell.vertices[0] = cell_vertices[c][0];
              cell.vertices[1] = cell_vertices[c][1];
              cell.vertices[2] = cell_vertices[c][2];
              cell.vertices[3] = cell_vertices[c][3];
              cell.material_id = (c < 4) ? ep.moderator_material : mat_id;
              cells.push_back(cell);
            }
          } else {
            CellData<dim> cell;
            cell.material_id = mat_id;

            if constexpr (dim == 1) {
              cell.vertices[0] = get_or_create_vertex(i * pin_pitch_x, 0, 0); // Left
              cell.vertices[1] = get_or_create_vertex((i+1) * pin_pitch_x, 0, 0); // Right
            }
            else if constexpr (dim == 2) {
              cell.vertices[0] = get_or_create_vertex(i * pin_pitch_x, j * pin_pitch_y, 0); // Bottom-Left
              cell.vertices[1] = get_or_create_vertex((i+1) * pin_pitch_x, j * pin_pitch_y, 0); // Bottom-Right
              cell.vertices[2] = get_or_create_vertex(i * pin_pitch_x, (j+1) * pin_pitch_y, 0); // Top-Left
              cell.vertices[3] = get_or_create_vertex((i+1) * pin_pitch_x, (j+1) * pin_pitch_y, 0); // Top-Right
            } 
            else if constexpr (dim == 3) {
              cell.vertices[0] = get_or_create_vertex(i * pin_pitch_x, j * pin_pitch_y, k * assembly_height);
              cell.vertices[1] = get_or_create_vertex((i+1) * pin_pitch_x, j * pin_pitch_y, k * assembly_height);
              cell.vertices[2] = get_or_create_vertex(i * pin_pitch_x, (j+1) * pin_pitch_y, k * assembly_height);
              cell.vertices[3] = get_or_create_vertex((i+1) * pin_pitch_x, (j+1) * pin_pitch_y, k * assembly_height);
              cell.vertices[4] = get_or_create_vertex(i * pin_pitch_x, j * pin_pitch_y, (k+1) * assembly_height);
              cell.vertices[5] = get_or_create_vertex((i+1) * pin_pitch_x, j * pin_pitch_y, (k+1) * assembly_height);
              cell.vertices[6] = get_or_create_vertex(i * pin_pitch_x, (j+1) * pin_pitch_y, (k+1) * assembly_height);
              cell.vertices[7] = get_or_create_vertex((i+1) * pin_pitch_x, (j+1) * pin_pitch_y, (k+1) * assembly_height);
            }

            cells.push_back(cell);
          }
        }
      }
    }

    if (geometry_data.get_explicit_pins_data().enabled) {
      GridTools::consistently_order_cells(cells);
    }

    SubCellData subcell_data;
    triangulation->create_triangulation(vertices, cells, subcell_data);

    if (geometry_data.get_explicit_pins_data().enabled) {
      const auto &ep = geometry_data.get_explicit_pins_data();
      const unsigned int pin_manifold_offset = 100;
      
      for (auto &cell : triangulation->active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        if (cell->material_id() != ep.moderator_material) {
          Point<dim> c = cell->center();
          int best_pin = -1;
          double min_dist = 1e10;
          for (unsigned int p = 0; p < explicit_pin_centers.size(); ++p) {
            double d = c.distance(explicit_pin_centers[p]);
            if (d < min_dist) {
              min_dist = d;
              best_pin = p;
            }
          }
          if (best_pin != -1 && min_dist > 0.1 * ep.radius) {
            cell->set_all_manifold_ids(pin_manifold_offset + best_pin);
          }
        }
      }

      for (unsigned int p = 0; p < explicit_pin_centers.size(); ++p) {
        triangulation->set_manifold(pin_manifold_offset + p, SphericalManifold<dim>(explicit_pin_centers[p]));
      }
    }

    // Set boundary ids
    for (auto &cell : triangulation->active_cell_iterators())
      for (const auto f : cell->face_indices())
        if (cell->face(f)->at_boundary()) 
          cell->face(f)->set_boundary_id(f);

    /* const Point<dim> bottom_left = Point<dim>();
    const Point<dim> upper_right =
      (dim == 2 ? Point<dim>(core_n_assemblies_x * rods_per_assembly_x * pin_pitch_x,
                             core_n_assemblies_y * rods_per_assembly_y * pin_pitch_y)
                : Point<dim>(core_n_assemblies_x * rods_per_assembly_x * pin_pitch_x,
                             core_n_assemblies_y * rods_per_assembly_y * pin_pitch_y,
                             core_n_assemblies_z * assembly_height));

    // Prepare subdivisions
    std::vector<unsigned int> n_subdivisions;
    n_subdivisions.emplace_back(core_n_assemblies_x * rods_per_assembly_x);
    n_subdivisions.emplace_back(core_n_assemblies_y * rods_per_assembly_y);
    if (dim >= 3)
      n_subdivisions.emplace_back(core_n_assemblies_z);

    // Create triangulation
    GridGenerator::subdivided_hyper_rectangle(*triangulation, n_subdivisions, bottom_left, upper_right, true);

    // Setup material ids
    for (auto &cell : triangulation->active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        const Point<dim> cell_center = cell->center();
  
        // tmp_x = global rod index ; ax = assembly index ; cx = local rod index
        const unsigned int tmp_x = int(cell_center[0] / pin_pitch_x);
        const unsigned int ax = tmp_x / rods_per_assembly_x;
        const unsigned int cx = tmp_x - ax * rods_per_assembly_x;
  
        const unsigned tmp_y = int(cell_center[1] / pin_pitch_y);
        const unsigned int ay = tmp_y / rods_per_assembly_y;
        const unsigned int cy = tmp_y - ay * rods_per_assembly_y;
  
        const unsigned int az = (dim == 2 ? 0 : int(cell_center[dim - 1] / assembly_height));
  
        Assert(ax < core_n_assemblies_x, ExcInternalError());
        Assert(ay < core_n_assemblies_y, ExcInternalError());
        Assert(az < core_n_assemblies_z, ExcInternalError());
        Assert(cx < rods_per_assembly_x, ExcInternalError());
        Assert(cy < rods_per_assembly_y, ExcInternalError());
  
        const unsigned int mapped_ay = (core_n_assemblies_y - 1) - ay;
        const unsigned int mapped_cy = (rods_per_assembly_y - 1) - cy;

        cell->set_material_id(geometry_data.get_assembly_pin(geometry_data.get_core_map(mapped_ay, ax, az), mapped_cy, cx));
      }
    } */
  }


  void NeutronSolver::setup_groups() {
    TimerOutput::Scope t(GlobalTimer::get(), "Setup groups");

    // Setup and collect all dof handlers and quadratures
    std::vector<const DoFHandler<dim>*> all_dof_handlers;
    std::vector<QGauss<1>> all_quadratures;
    for (const auto &group : energy_groups) {
      group->setup_dofs();
      all_dof_handlers.push_back(&(group->get_dof_handler()));
      all_quadratures.push_back(QGauss<1>(group->get_degree() + 1));
    }

    if (!mf_storage)
      mf_storage = std::make_shared<MatrixFree<dim, double>>();
    else
      mf_storage->clear();

    AffineConstraints<double> empty_constraints;
    empty_constraints.close();
    std::vector<const AffineConstraints<double>*> all_constraints(all_dof_handlers.size(), &empty_constraints);
  
    // Setup the mf storage with complete dof handlers
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values);
    additional_data.mapping_update_flags_inner_faces = (update_values | update_gradients | update_JxW_values | update_normal_vectors | update_inverse_jacobians);
    additional_data.mapping_update_flags_boundary_faces = (update_values | update_JxW_values);
    
    mf_storage->reinit(mapping, all_dof_handlers, all_constraints, all_quadratures, additional_data);

    // Setup systems
    for (const auto &group : energy_groups)
      group->setup_system(mf_storage);
  }


  double NeutronSolver::get_total_fission_source(bool is_adjoint) const {
    double local_fission = 0.0;
    for (const auto &group : energy_groups)
      local_fission += group->get_fission_source(is_adjoint);

    return Utilities::MPI::sum(local_fission, MPI_COMM_WORLD);
  }


  void NeutronSolver::solve_eigenvalue_problem(bool is_adjoint) {
    const double tol = 1e-8;
    const unsigned int maxiter = 500;
    double error = 1.0;
    unsigned int iter = 1;

    double adjoint_source_norm = k_eff;
    double adjoint_source_norm_old = k_eff_old;

    if (is_adjoint)
      for (const auto &group : energy_groups)
        group->copy_forward_to_adjoint();

    do {    
      if (is_adjoint) {
        // Inverse sweep (up-scattering is dominant)
        for (auto it = energy_groups.rbegin(); it != energy_groups.rend(); ++it) {
          (*it)->compute_rhs(energy_groups, is_adjoint);
          (*it)->solve(is_adjoint);
        }

        adjoint_source_norm = get_total_fission_source(is_adjoint);
        if (adjoint_source_norm > 1e-12)
          error = std::abs(adjoint_source_norm - adjoint_source_norm_old) / std::abs(adjoint_source_norm);
        adjoint_source_norm_old = adjoint_source_norm;
      }
      else {
        // Forward sweep (down-scattering is dominant)
        for (auto it = energy_groups.begin(); it != energy_groups.end(); ++it) {
          (*it)->compute_rhs(energy_groups, is_adjoint);
          (*it)->solve(is_adjoint);
        }

        // Update k_eff
        k_eff = get_total_fission_source(is_adjoint);
        if (k_eff > 1e-12)
          error = std::abs(k_eff - k_eff_old) / std::abs(k_eff);
        k_eff_old = k_eff;
      }

      // Update old_solution for all groups
      double current_k = is_adjoint ? adjoint_source_norm : k_eff;
      for (const auto &group : energy_groups) {
        group->update_solution(current_k, is_adjoint);
      }

      if (is_adjoint)
        pcout << "Iter " << std::setw(3) << std::right << iter 
              << " | adjoint_source_norm: " << std::fixed << std::setprecision(6) << adjoint_source_norm 
              << " | Adjoint norm err: " << std::scientific << error << std::endl;
      else
        pcout << "Iter " << std::setw(3) << std::right << iter 
              << " | k_eff: " << std::fixed << std::setprecision(6) << k_eff 
              << " | err: " << std::scientific << error << std::endl;

      ++iter;
    } while ((error > tol) && (iter < maxiter));
  }


  void NeutronSolver::compute_weighted_error(ErrorEstimates &estimates) const {
    TimerOutput::Scope t(GlobalTimer::get(), "NeutronSolver::EstimateError");

    estimates.global_group_errors.resize(energy_groups.size(), 0.0);
    estimates.h_refinement_estimators.reinit(triangulation->n_active_cells());

    for (unsigned int g = 0; g < energy_groups.size(); ++g) {
      auto &group = energy_groups[g];

      Vector<float> primal_kelly(triangulation->n_active_cells());
      Vector<float> dual_kelly(triangulation->n_active_cells());

      QGauss<dim - 1> face_quadrature(group->get_degree() + 1);

      // Index sets
      const IndexSet locally_owned_dofs = group->get_dof_handler().locally_owned_dofs();
      const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(group->get_dof_handler());

      // Get the locally relevant solution of the zero mode
      LinearAlgebra::distributed::Vector<double> relevant_diffusion;
      relevant_diffusion.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
      relevant_diffusion = group->get_solution().block(0);
      relevant_diffusion.update_ghost_values();

      // Get the locally relevant adjoint solution of the zero mode
      LinearAlgebra::distributed::Vector<double> relevant_adjoint_diffusion;
      relevant_adjoint_diffusion.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
      relevant_adjoint_diffusion = group->get_adjoint_solution().block(0);
      relevant_adjoint_diffusion.update_ghost_values();

      KellyErrorEstimator<dim>::estimate(
        group->get_dof_handler(),
        face_quadrature,
        {},
        relevant_diffusion,
        primal_kelly
      );

      /* KellyErrorEstimator<dim>::estimate(
        group->get_dof_handler(),
        face_quadrature,
        {},
        relevant_adjoint_diffusion,
        dual_kelly
      ); */

      float local_group_error_sum = 0.0;
      for (const auto &cell : triangulation->active_cell_iterators()) {
        if (cell->is_locally_owned()) {
          const unsigned int index = cell->active_cell_index();
          float goal_oriented_cell_error = primal_kelly(index); // * dual_kelly(index);
          local_group_error_sum += goal_oriented_cell_error;

          // h refinement if the group is thermic (most 2 thermic groups)
          if (g >= energy_groups.size() - 2)
            estimates.h_refinement_estimators(index) += goal_oriented_cell_error;
        }
      }
      
      estimates.global_group_errors[g] = Utilities::MPI::sum(local_group_error_sum, MPI_COMM_WORLD);
    }
  }


  void NeutronSolver::run() {

    {
      const unsigned int n_mpi_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      const unsigned int n_vect_doubles = VectorizedArray<double>::size();
      const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;
      
      pcout << "===========================================" << std::endl;
      pcout << "RUNNING BENCHMARK " << bench_name << std::endl;
      pcout << "Running with " << n_mpi_procs << " MPI process(es)" << std::endl;
      pcout << "Vectorization over " << n_vect_doubles << " doubles = " << n_vect_bits << " bits (" << Utilities::System::get_current_vectorization_level() << ')' << std::endl;
      pcout << "===========================================" << std::endl << std::endl << std::endl;
    }

    init_mesh();
    triangulation->refine_global(1);

    k_eff = 1.0;
    k_eff_old = k_eff;

    // Create energy groups with degree 1
    std::vector<unsigned int> group_degrees(material_data.get_n_groups(), 1);
    for (unsigned int group = 0; group < material_data.get_n_groups(); ++group) 
      energy_groups.emplace_back(std::make_unique<solver::EnergyGroup<dim>>(group, group_degrees[group], material_data, geometry_data, triangulation, mapping));

    const unsigned int n_cycles = 5;
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle) {
      pcout << "===========================================" << std::endl;
      pcout << "Starting cycle " << cycle << ':' << std::endl;
      pcout << "===========================================" << std::endl;
      pcout << "Number of active cells: " << triangulation->n_global_active_cells() << std::endl;

      setup_groups();
      
      // Solve forward problem
      {
        TimerOutput::Scope t(GlobalTimer::get(), "NeutronSolver::Solve::forward");
        solve_eigenvalue_problem(false);
      }

      // Cycles are finished, we don't need more refinement
      if (cycle == n_cycles - 1) break; 
    
      // Solve adjoint problem
      /* {
        TimerOutput::Scope t(GlobalTimer::get(), "NeutronSolver::Solve::adjoint");
        solve_eigenvalue_problem(true);
      } */

      // Estimate errors
      ErrorEstimates errors;
      compute_weighted_error(errors);

      GridRefinement::refine_and_coarsen_fixed_number(
        *triangulation,
        errors.h_refinement_estimators,
        0.3,
        0.03
      );

      for (const auto &group : energy_groups)
        group->prepare_h_transfer();

      triangulation->execute_coarsening_and_refinement();
      // triangulation->refine_global(1);

      // h-refinement and p-refinement evaluation
      float max_error = *std::max_element(errors.global_group_errors.begin(), errors.global_group_errors.end());
      for (unsigned int g = 0; g < energy_groups.size(); ++g) {
        energy_groups[g]->execute_h_transfer();

        float threshold = 0.5f * static_cast<float>(std::pow((g + 1.0) / (energy_groups.size() + 1.0), 1.0)) * max_error;
        pcout << "Group " << g << " Adjoint-Weighted Error: " << errors.global_group_errors[g] << " | Threshold: " << threshold << std::endl;
        
        if (errors.global_group_errors[g] > threshold && energy_groups[g]->get_degree() < max_degree && false) { 
          pcout << "  -> Increasing p-degree to " << energy_groups[g]->get_degree() + 1 << std::endl;
          energy_groups[g]->set_degree(energy_groups[g]->get_degree() + 1);
        }
      }

      GlobalTimer::get().print_summary();
      GlobalTimer::get().reset();
    }

    // Print results
    for (unsigned int group = 0; group < material_data.get_n_groups(); ++group)
      energy_groups[group]->output_results(bench_name);
    
  }

}