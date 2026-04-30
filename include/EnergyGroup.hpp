#pragma once

#include "MaterialData.hpp"
#include "GeometryData.hpp"
#include "CrossSectionManager.hpp"

#include "ZeroModeOperator.hpp"
#include "SecondModeOperator.hpp"
#include "CouplingOperator.hpp"
#include "InverseOperator.hpp"
#include "SP3Operator.hpp"
#include "MultigridPreconditioner.hpp"
#include "BlockGSPreconditioner.hpp"
#include "BlockDiagonalPreconditioner.hpp"
#include "TransposeWrapper.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <memory>
#include <vector>


namespace solver {
  
  template <unsigned int dim>
  class EnergyGroup {
  public:
    template <typename number> using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;
    template <typename number> using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    EnergyGroup(const unsigned int group,
                const unsigned int p_degree,
                const data::MaterialData &material_data,
                const data::GeometryData &geometry_data,
                std::shared_ptr<dealii::Triangulation<dim>> triangulation,
                const dealii::MappingQ1<dim> &mapping
    ) :
      group(group),
      p_degree(p_degree),
      material_data(material_data),
      geometry_data(geometry_data),
      triangulation(triangulation),
      mapping(mapping),
      fe(std::make_unique<dealii::FE_DGQ<dim>>(p_degree)),
      dof_handler(*triangulation),

      solution(2),
      solution_old(2),
      adjoint_solution(2),
      adjoint_solution_old(2),
      system_rhs(2)
    {}

    void setup_dofs();
    void setup_system(std::shared_ptr<const dealii::MatrixFree<dim, double>>);
  
    void compute_rhs(const std::vector<std::unique_ptr<EnergyGroup<dim>>> &, bool);
    void solve(bool);
    double get_fission_source(bool) const;
    
    void update_solution(const double, bool);
    void copy_forward_to_adjoint();
    void output_results(const std::string &) const;
    
    void set_degree(unsigned int);
    void prepare_h_transfer();
    void execute_h_transfer();
    
    unsigned int get_degree() const;
    const dealii::DoFHandler<dim>& get_dof_handler() const { return dof_handler; }
    BlockVectorType<double> get_solution() const;
    BlockVectorType<double> get_adjoint_solution() const;


  private:
    void setup_multigrid();
    void setup_coefficients(std::shared_ptr<const dealii::MatrixFree<dim, double>>);
    void setup_feevals();

    const unsigned int group;
    unsigned int p_degree;
    const data::MaterialData &material_data;
    const data::GeometryData &geometry_data;

    std::shared_ptr<dealii::Triangulation<dim>> triangulation;
    const dealii::MappingQ1<dim> &mapping;
    
    std::unique_ptr<dealii::FE_DGQ<dim>> fe;
    dealii::DoFHandler<dim> dof_handler;
  
    bool needs_p_transfer = false;
    std::unique_ptr<dealii::FE_DGQ<dim>> transfer_old_fe;
    std::unique_ptr<dealii::DoFHandler<dim>> transfer_old_dof_handler;

    // h-refinement
    std::unique_ptr<BlockVectorType<double>> h_interpolated_solution;
    std::unique_ptr<dealii::SolutionTransfer<dim, VectorType<double>>> sol_transfer;

    // Inner operators
    std::shared_ptr<ZeroModeOperator<dim, double>> zero_mode_operator;
    std::shared_ptr<SecondModeOperator<dim, double>> second_mode_operator;
    std::shared_ptr<CouplingOperator<dim, double>> coupling_operator;

    // Inner preconditioners
    using InnerPreconditionerZero = MultigridPreconditioner<dim, float, ZeroModeOperator<dim, float>>;
    using InnerPrecontionerSecond = MultigridPreconditioner<dim, float, SecondModeOperator<dim, float>>;
    // using InnerPrecontionerSecond = dealii::PreconditionChebyshev<SecondModeOperator<dim, double>, VectorType<double>>;
    std::shared_ptr<InnerPreconditionerZero> inner_preconditioner_zero;
    std::shared_ptr<InnerPrecontionerSecond> inner_preconditioner_second;

    // Inner solvers
    using InnerSolverZero = InverseOperator<VectorType<double>, ZeroModeOperator<dim, double>, InnerPreconditionerZero>;
    using InnerSolverSecond = InverseOperator<VectorType<double>, SecondModeOperator<dim, double>, InnerPrecontionerSecond>;
    std::shared_ptr<InnerSolverZero> inner_solver_zero;
    std::shared_ptr<InnerSolverSecond> inner_solver_second;
    
    // Global operator and preconditioner
    using Preconditioner = BlockGSPreconditioner<BlockVectorType<double>, InnerSolverZero, InnerSolverSecond, CouplingOperator<dim, double>>;
    using BlockDiagPreconditioner = BlockDiagonalPreconditioner<BlockVectorType<double>, InnerPreconditionerZero, InnerPrecontionerSecond>;
    
    std::shared_ptr<SP3Operator<dim, double>> sp3_operator;
    std::shared_ptr<Preconditioner> preconditioner;
    std::shared_ptr<BlockDiagPreconditioner> block_diag_preconditioner;

    BlockVectorType<double> solution;
    BlockVectorType<double> solution_old;
    BlockVectorType<double> adjoint_solution;
    BlockVectorType<double> adjoint_solution_old;
    BlockVectorType<double> system_rhs;

    // Coefficients managers
    std::shared_ptr<CrossSectionManager<dim, double>> material_manager;
    std::shared_ptr<CrossSectionManager<dim, float>> mg_material_manager;
  
    // FEEvaluation (to compute rhs)
    std::unique_ptr<dealii::FEEvaluation<dim, -1, 0, 1, double>> phi0;
    std::unique_ptr<dealii::FEEvaluation<dim, -1, 0, 1, double>> phi2;
    std::vector<std::unique_ptr<dealii::FEEvaluation<dim, -1, 0, 1, double>>> phi_prime_old;
    std::vector<std::unique_ptr<dealii::FEEvaluation<dim, -1, 0, 1, double>>> phi_prime;

  };

}