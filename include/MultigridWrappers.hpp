#pragma once

#include "SP3Operator.hpp"

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
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <array>
#include <vector>
#include <memory>
#include <type_traits>


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


  template <unsigned int dim, typename number, typename OperatorType>
  class CoarseMatrixBuilder {
  public:
    // Used with ZeroModeOperator and SecondModeOperator
    template <typename T = OperatorType, typename std::enable_if<!std::is_same<T, SP3Operator<dim, number>>::value, int>::type = 0>
    static void build(const dealii::DoFHandler<dim> &dof_handler, const OperatorType &level_operator, dealii::TrilinosWrappers::SparseMatrix &matrix) {
      const dealii::IndexSet locally_owned_level_dofs = dof_handler.locally_owned_mg_dofs(0);
      dealii::TrilinosWrappers::SparsityPattern dsp(locally_owned_level_dofs, MPI_COMM_WORLD);
      dealii::MGTools::make_flux_sparsity_pattern(dof_handler, dsp, 0);
      dsp.compress();
      
      matrix.reinit(dsp);
      level_operator.compute_matrix(matrix);
      matrix.compress(dealii::VectorOperation::add);
    }

    // Used with SP3Operator
    static void build(const dealii::DoFHandler<dim> &dof_handler, const SP3Operator<dim, number> &level_operator, dealii::TrilinosWrappers::SparseMatrix &matrix) {
      level_operator.compute_matrix(dof_handler, matrix);
    }
  };


  template <unsigned int dim, typename number>
  class SP3BlockMGTransfer : public dealii::MGTransferBase<dealii::LinearAlgebra::distributed::BlockVector<number>> {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    void build(const dealii::DoFHandler<dim> &dof_handler, const std::vector<std::shared_ptr<const dealii::Utilities::MPI::Partitioner>> &external_partitioners) {
      scalar_transfer.build(dof_handler, external_partitioners);

      const unsigned int max_level = dof_handler.get_triangulation().n_global_levels() - 1;
      for (unsigned int block = 0; block < n_blocks; ++block) {
        copy_to_mg_scratch[block].resize(0, max_level);
        copy_from_mg_scratch[block].resize(0, max_level);

        for (unsigned int level = 0; level <= max_level; ++level) {
          if (level < external_partitioners.size() && external_partitioners[level]) {
            copy_to_mg_scratch[block][level].reinit(external_partitioners[level]);
            copy_from_mg_scratch[block][level].reinit(external_partitioners[level]);
          }
        }
      }
    }

    void prolongate(const unsigned int to_level, BlockVectorType &dst, const BlockVectorType &src) const override {
      for (unsigned int block = 0; block < n_blocks; ++block)
        scalar_transfer.prolongate(to_level, dst.block(block), src.block(block));
    }

    void prolongate_and_add(const unsigned int to_level, BlockVectorType &dst, const BlockVectorType &src) const override {
      for (unsigned int block = 0; block < n_blocks; ++block)
        scalar_transfer.prolongate_and_add(to_level, dst.block(block), src.block(block));
    }

    void restrict_and_add(const unsigned int from_level, BlockVectorType &dst, const BlockVectorType &src) const override {
      for (unsigned int block = 0; block < n_blocks; ++block)
        scalar_transfer.restrict_and_add(from_level, dst.block(block), src.block(block));
    }

    template <typename BlockVectorType2>
    void copy_to_mg(const dealii::DoFHandler<dim> &dof_handler, dealii::MGLevelObject<BlockVectorType> &dst, const BlockVectorType2 &src) const {
      for (unsigned int block = 0; block < n_blocks; ++block)
        scalar_transfer.copy_to_mg(dof_handler, copy_to_mg_scratch[block], src.block(block));

      for (unsigned int level = dst.min_level(); level <= dst.max_level(); ++level) {
        if (dst[level].n_blocks() != n_blocks)
          dst[level].reinit(n_blocks);

        for (unsigned int block = 0; block < n_blocks; ++block) {
          ensure_layout(dst[level].block(block), copy_to_mg_scratch[block][level]);
          dst[level].block(block) = copy_to_mg_scratch[block][level];
        }

        dst[level].collect_sizes();
      }
    }

    template <typename BlockVectorType2>
    void copy_from_mg(const dealii::DoFHandler<dim> &dof_handler, BlockVectorType2 &dst, const dealii::MGLevelObject<BlockVectorType> &src) const {
      for (unsigned int block = 0; block < n_blocks; ++block) {
        copy_block_to_scalar_mg(block, src);
        scalar_transfer.copy_from_mg(dof_handler, dst.block(block), copy_from_mg_scratch[block]);
      }
    }

    template <typename BlockVectorType2>
    void copy_from_mg_add(const dealii::DoFHandler<dim> &dof_handler, BlockVectorType2 &dst, const dealii::MGLevelObject<BlockVectorType> &src) const {
      for (unsigned int block = 0; block < n_blocks; ++block) {
        copy_block_to_scalar_mg(block, src);
        scalar_transfer.copy_from_mg_add(dof_handler, dst.block(block), copy_from_mg_scratch[block]);
      }
    }

  private:
    static constexpr unsigned int n_blocks = 2; // TODO create a global struct SP3Traits and use n_modes=2 everywhere in the code instead of hardcode 2

    void ensure_layout(VectorType &vector, const VectorType &prototype) const {
      if (vector.size() != prototype.size() || vector.locally_owned_size() != prototype.locally_owned_size() || vector.get_partitioner().get() != prototype.get_partitioner().get())
        vector.reinit(prototype, true);
    }

    void copy_block_to_scalar_mg(const unsigned int block, const dealii::MGLevelObject<BlockVectorType> &src) const {
      for (unsigned int level = src.min_level(); level <= src.max_level(); ++level) {
        ensure_layout(copy_from_mg_scratch[block][level], src[level].block(block));
        copy_from_mg_scratch[block][level] = src[level].block(block);
      }
    }

    dealii::MGTransferMatrixFree<dim, number> scalar_transfer;
    mutable std::array<dealii::MGLevelObject<VectorType>, n_blocks> copy_to_mg_scratch;
    mutable std::array<dealii::MGLevelObject<VectorType>, n_blocks> copy_from_mg_scratch;
  };

}
