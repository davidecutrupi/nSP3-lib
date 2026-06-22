#pragma once


#include <deal.II/base/exception_macros.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/multigrid/mg_base.h>

#include <vector>
#include <memory>


namespace solver {

  template <typename VectorType, typename OperatorType>
  class LevelMatrixWrapper : public dealii::MGMatrixBase<VectorType> {
  public:
    LevelMatrixWrapper(const std::vector<std::shared_ptr<OperatorType>> &ops) : operators(ops) {}

    unsigned int get_minlevel() const override { return 0; }
    unsigned int get_maxlevel() const override { return operators.empty() ? 0 : operators.size() - 1; }
    void vmult(const unsigned int level, VectorType &dst, const VectorType &src) const override { operators[level]->vmult(dst, src); }
    void vmult_add(const unsigned int level, VectorType &dst, const VectorType &src) const override { operators[level]->vmult_add(dst, src); }
    void Tvmult(const unsigned int level, VectorType &dst, const VectorType &src) const override { operators[level]->Tvmult(dst, src); }
    void Tvmult_add(const unsigned int level, VectorType &dst, const VectorType &src) const override { operators[level]->Tvmult_add(dst, src); }

  private:
    std::vector<std::shared_ptr<OperatorType>> operators;
  };


  template <typename VectorType, typename SmootherType>
  class LevelSmootherWrapper : public dealii::MGSmootherBase<VectorType> {
  public:
    LevelSmootherWrapper() = default;
    void initialize(const dealii::MGLevelObject<SmootherType>* smoothers_ptr) { this->smoothers = smoothers_ptr; }
    void smooth(const unsigned int level, VectorType &u, const VectorType &rhs) const override { (*smoothers)[level].step(u, rhs); }
    void clear() override { smoothers = nullptr; }

  private:
    const dealii::MGLevelObject<SmootherType> *smoothers = nullptr;
  };


  template <unsigned int dim, typename VectorType>
  class MGCoarseGridTrilinosWrapper : public dealii::MGCoarseGridBase<VectorType> {
  public:
    MGCoarseGridTrilinosWrapper() = default;

    void initialize(const dealii::TrilinosWrappers::SparseMatrix &coarse_matrix) {
      dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
      data.solver_type = "Amesos_Klu"; 
      direct_solver.initialize(coarse_matrix, data);
    }

    void initialize(const dealii::TrilinosWrappers::SparseMatrix &coarse_matrix, const dealii::DoFHandler<dim> &) {
      initialize(coarse_matrix);
    }

    virtual void operator()(const unsigned int level, VectorType &dst, const VectorType &src) const override {
      (void) level;
      if (temp_src_double.size() == 0) {
        temp_src_double.reinit(src, true);
        temp_dst_double.reinit(dst, true);
      }

      temp_src_double = src;
      direct_solver.vmult(temp_dst_double, temp_src_double);
      dst = temp_dst_double;
    }

  private:
    dealii::TrilinosWrappers::SolverDirect direct_solver;
    // This solver only supports doubles
    mutable dealii::LinearAlgebra::distributed::Vector<double> temp_src_double;
    mutable dealii::LinearAlgebra::distributed::Vector<double> temp_dst_double;
  };


  template <unsigned int dim, typename number>
  class MGCoarseGridTrilinosBlockWrapper : public dealii::MGCoarseGridBase<dealii::LinearAlgebra::distributed::BlockVector<number>> {
  public:
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    void initialize(const dealii::TrilinosWrappers::SparseMatrix &coarse_matrix, const dealii::DoFHandler<dim> &dof_handler) {
      scalar_size = dof_handler.n_dofs(0);
      const dealii::IndexSet scalar_owned = dof_handler.locally_owned_mg_dofs(0);

      dealii::IndexSet monolithic_owned(2 * scalar_size);
      for (const auto index : scalar_owned) {
        monolithic_owned.add_index(index);
        monolithic_owned.add_index(index + scalar_size);
      }
      monolithic_owned.compress();

      temp_src_double.reinit(monolithic_owned, MPI_COMM_WORLD);
      temp_dst_double.reinit(monolithic_owned, MPI_COMM_WORLD);

      dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
      data.solver_type = "Amesos_Mumps";
      direct_solver.initialize(coarse_matrix, data);
    }

    void operator()(const unsigned int level, BlockVectorType &dst, const BlockVectorType &src) const override {
      (void) level;
      AssertDimension(src.n_blocks(), 2);
      AssertDimension(dst.n_blocks(), 2);

      const auto n_local = src.block(0).locally_owned_size();

      AssertDimension(src.block(1).locally_owned_size(), n_local);
      AssertDimension(temp_src_double.locally_owned_size(), 2 * n_local);

      // Fast copy onto monolhitic vector
      double *start = temp_src_double.begin();
      for (dealii::types::global_dof_index i = 0; i < n_local; ++i) {
        start[i] = static_cast<double>(src.block(0).local_element(i));
        start[i+n_local] = static_cast<double>(src.block(1).local_element(i));
      }

      // Solve
      direct_solver.vmult(temp_dst_double, temp_src_double);

      // Fast copy onto the dst block vector
      start = temp_dst_double.begin();
      for (dealii::types::global_dof_index i = 0; i < n_local; ++i) {
        dst.block(0).local_element(i) = static_cast<number>(start[i]);
        dst.block(1).local_element(i) = static_cast<number>(start[i + n_local]);
      }
    }

  private:
    dealii::types::global_dof_index scalar_size = 0;
    dealii::TrilinosWrappers::SolverDirect direct_solver;
    mutable dealii::TrilinosWrappers::MPI::Vector temp_src_double;
    mutable dealii::TrilinosWrappers::MPI::Vector temp_dst_double;
  };
