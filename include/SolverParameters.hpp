#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/types.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <stdexcept>
#include <string>


namespace solver {

  class SolverParameters {
  public:
  
    SolverParameters() {
      dealii::ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input("../params.prm");
      parse_parameters(prm);
      normalize_and_validate();
    }

    std::string benchmark = "data";
    std::string output_directory = "../out";
    std::string output_prefix = "data";
    std::string fe_type = "DG";
    std::string h_ref_type = "none";
    bool output_power_distribution = true;
    std::string power_quantity = "fission source";

    unsigned int n_cycles = 2;
    unsigned int max_p_degree = 5;

    unsigned int eigen_max_iterations = 500;
    double eigen_tolerance = 1e-8;

    unsigned int group_max_iterations = 1000;
    double group_tolerance = 1e-8;

    dealii::types::global_dof_index coarse_direct_klu_max_dofs = 10000;
    dealii::types::global_dof_index coarse_p_coarsening_min_dofs = 75000;
    unsigned int coarse_p_coarsening_min_degree = 3;
    std::string coarse_p_coarsening_sequence_string = "bisect";
    dealii::MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType coarse_p_coarsening_sequence = dealii::MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::bisect;


  private:
    void declare_parameters(dealii::ParameterHandler &prm) const {
      prm.declare_entry("Benchmark", benchmark, dealii::Patterns::Anything(), "Benchmark name used as input path and default output prefix.");
      prm.declare_entry("Output directory", output_directory, dealii::Patterns::Anything(), "Directory for VTU output files. The solver does not create this directory.");
      prm.declare_entry("FE type", fe_type, dealii::Patterns::Selection("DG|CG"), "Finite element family.");
      prm.declare_entry("Output power distribution", output_power_distribution ? "true" : "false", dealii::Patterns::Bool(), "Whether to write the final pin-power CSV.");
      prm.declare_entry("Power Quantity", power_quantity, dealii::Patterns::Selection("fission source|fission rate"), "Pin-power quantity: fission source uses nu_sigma_f; fission rate uses sigma_f.");
      
      prm.declare_entry("Refinement cycles", std::to_string(n_cycles), dealii::Patterns::Integer(1), "Number of solve/adapt cycles.");
      prm.declare_entry("Max polynomial degree", std::to_string(max_p_degree), dealii::Patterns::Integer(1), "Maximum group-global p degree used by adaptive p-refinement.");
      prm.declare_entry("Spatial Refinement Type", h_ref_type, dealii::Patterns::Selection("global|adaptive|none"), "Type of spatial refinement: adaptive, global.");

      prm.enter_subsection("Eigenvalue solver");
      prm.declare_entry("Max iterations", std::to_string(eigen_max_iterations), dealii::Patterns::Integer(1), "Maximum outer eigenvalue iterations.");
      prm.declare_entry("Tolerance", "1e-8", dealii::Patterns::Double(0.0), "Relative tolerance for the outer eigenvalue iteration.");
      prm.leave_subsection();

      prm.enter_subsection("Group solver");
      prm.declare_entry("Max iterations", std::to_string(group_max_iterations), dealii::Patterns::Integer(1), "Maximum per-energy-group Krylov iterations.");
      prm.declare_entry("Tolerance", "1e-8", dealii::Patterns::Double(0.0), "Per-energy-group Krylov tolerance.");
      prm.leave_subsection();

      prm.enter_subsection("Multigrid");
      prm.declare_entry("Coarse direct KLU max dofs", std::to_string(coarse_direct_klu_max_dofs), dealii::Patterns::Integer(0), "Use Amesos_Klu for coarse matrices below this monolithic size; use Amesos_Mumps at or above it.");
      prm.declare_entry("Coarse p-coarsening min dofs", std::to_string(coarse_p_coarsening_min_dofs), dealii::Patterns::Integer(0), "Enable p-coarsened coarse levels when the h-coarsest monolithic matrix exceeds this size.");
      prm.declare_entry("Coarse p-coarsening min degree", std::to_string(coarse_p_coarsening_min_degree), dealii::Patterns::Integer(1), "Minimum active polynomial degree for adding p-coarsened coarse levels.");
      prm.declare_entry("Coarse p-coarsening sequence", coarse_p_coarsening_sequence_string, dealii::Patterns::Selection("bisect|decrease_by_one|go_to_one"), "Polynomial coarsening sequence for extra coarse levels.");
      prm.leave_subsection();
    }

    void parse_parameters(dealii::ParameterHandler &prm) {
      benchmark = prm.get("Benchmark");
      output_directory = prm.get("Output directory");
      fe_type = prm.get("FE type");
      output_power_distribution = prm.get_bool("Output power distribution");
      power_quantity = prm.get("Power Quantity");
      
      n_cycles = static_cast<unsigned int>(prm.get_integer("Refinement cycles"));
      max_p_degree = static_cast<unsigned int>(prm.get_integer("Max polynomial degree"));
      h_ref_type = prm.get("Spatial Refinement Type");

      prm.enter_subsection("Eigenvalue solver");
      eigen_max_iterations = static_cast<unsigned int>(prm.get_integer("Max iterations"));
      eigen_tolerance = prm.get_double("Tolerance");
      prm.leave_subsection();

      prm.enter_subsection("Group solver");
      group_max_iterations = static_cast<unsigned int>(prm.get_integer("Max iterations"));
      group_tolerance = prm.get_double("Tolerance");
      prm.leave_subsection();

      prm.enter_subsection("Multigrid");
      coarse_direct_klu_max_dofs = static_cast<dealii::types::global_dof_index>(prm.get_integer("Coarse direct KLU max dofs"));
      coarse_p_coarsening_min_dofs = static_cast<dealii::types::global_dof_index>(prm.get_integer("Coarse p-coarsening min dofs"));
      coarse_p_coarsening_min_degree = static_cast<unsigned int>(prm.get_integer("Coarse p-coarsening min degree"));
      coarse_p_coarsening_sequence_string = prm.get("Coarse p-coarsening sequence");
      prm.leave_subsection();
    }

    void normalize_and_validate() {
      if (benchmark.empty()) throw std::runtime_error("SolverParameters: Benchmark name must not be empty.");
      if (output_directory.empty()) output_directory = ".";
      if (fe_type != "DG" && fe_type != "CG") throw std::runtime_error("SolverParameters: FE type must be either DG or CG.");
      if (power_quantity != "fission source" && power_quantity != "fission rate") throw std::runtime_error("SolverParameters: Power Quantity must be either 'fission source' or 'fission rate'.");

      if (coarse_p_coarsening_sequence_string == "bisect") coarse_p_coarsening_sequence = dealii::MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::bisect;
      else if (coarse_p_coarsening_sequence_string == "decrease_by_one") coarse_p_coarsening_sequence = dealii::MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::decrease_by_one;
      else if (coarse_p_coarsening_sequence_string == "go_to_one") coarse_p_coarsening_sequence = dealii::MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::go_to_one;
      else throw std::runtime_error("SolverParameters: invalid multigrid p-coarsening sequence.");
    }

  };

}
