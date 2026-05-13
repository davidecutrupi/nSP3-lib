#include "GlobalTimer.hpp"
#include "NeutronSolver.hpp"


int main(int argc, char **argv) {
  try {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    std::string bench_name = argc == 1 ? "data" : argv[1];
    solver::NeutronSolver neutron_solver(bench_name);
    neutron_solver.run();

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