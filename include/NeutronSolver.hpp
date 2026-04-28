#pragma once

#include "GeometryData.hpp"
#include "GlobalTimer.hpp"
#include "MaterialData.hpp"
#include "EnergyGroup.hpp"

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/lac/la_parallel_block_vector.h>


namespace solver {

  struct ErrorEstimates {
    std::vector<float> global_group_errors;
    dealii::Vector<float> h_refinement_estimators; 
  };

  constexpr unsigned int dim = 2;
  constexpr unsigned int max_degree = 5;

  class NeutronSolver {
  public:
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<double>;
    
    NeutronSolver(const std::string &bench_name) :
      bench_name(bench_name),
      material_data(data::MaterialData::from_file("../benchmarks/"  + bench_name + ".json")),
      geometry_data(data::GeometryData::from_file("../benchmarks/"  + bench_name + ".json")),
      triangulation(
        MPI_COMM_WORLD,
        dealii::Triangulation<dim>::limit_level_difference_at_vertices,
        dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      mpi_size(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
      mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
      pcout(std::cout, mpi_rank == 0)
    {
      GlobalTimer::init(MPI_COMM_WORLD, pcout);
    };
    void run();

  private:
    void init_mesh();
    void setup_groups();
    void solve_eigenvalue_problem(bool);
    double get_total_fission_source(bool) const;
    void compute_weighted_error(ErrorEstimates &) const;

    const std::string &bench_name;
    const data::MaterialData material_data;
    const data::GeometryData geometry_data;
    
    dealii::parallel::distributed::Triangulation<dim> triangulation;
    const dealii::MappingQ1<dim> mapping;

    std::shared_ptr<dealii::MatrixFree<dim, double>> mf_storage;

    std::vector<std::unique_ptr<EnergyGroup<dim>>> energy_groups;
    double k_eff_old;
    double k_eff;

    const unsigned int mpi_size;
    const unsigned int mpi_rank;
    dealii::ConditionalOStream pcout;

  }; 

}