#include "GlobalTimer.hpp"
#include "NeutronSolver.hpp"

#include <stdexcept>
#include <string>


int main(int argc, char **argv) {
  try {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    {
      const solver::SolverParameters parameters;
      const std::string benchmark_path = "../benchmarks/" + parameters.benchmark + ".json";
      const unsigned int dimension = data::GeometryData::read_dimension_from_file(benchmark_path);

      if (dimension == 1) {
        solver::NeutronSolver<1u> neutron_solver(benchmark_path, parameters);
        neutron_solver.run();
      }
      else if (dimension == 2) {
        solver::NeutronSolver<2u> neutron_solver(benchmark_path, parameters);
        neutron_solver.run();
      }
      else if (dimension == 3) {
        solver::NeutronSolver<3u> neutron_solver(benchmark_path, parameters);
        neutron_solver.run();
      }
      else
        throw std::runtime_error("Unsupported dimension");
    }

    GlobalTimer::clear();
  }

  catch (std::exception &exc) {
    std::cerr << std::endl
      << std::endl
      << "----------------------------------------------------"
      << std::endl;
    std::cerr << "Exception on processing: " << std::endl
      << exc.what() << std::endl
      << "Aborting!" << std::endl
      << "----------------------------------------------------"
      << std::endl;

    return 1;
  }

  catch (...) {
    std::cerr << std::endl
      << std::endl
      << "----------------------------------------------------"
      << std::endl;
    std::cerr << "Unknown exception!" << std::endl
      << "Aborting!" << std::endl
      << "----------------------------------------------------"
      << std::endl;
    return 1;
  }
 
  return 0;
}
