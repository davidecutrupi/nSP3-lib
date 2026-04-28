#pragma once

#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/template_constraints.h>

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

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


  template <typename VectorType>
  class MGCoarseGridTrilinosWrapper : public dealii::MGCoarseGridBase<VectorType> {
  public:
    MGCoarseGridTrilinosWrapper() = default;

    void initialize(const dealii::TrilinosWrappers::SparseMatrix &coarse_matrix) {
      dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
      data.solver_type = "Amesos_Klu"; 
      direct_solver.initialize(coarse_matrix, data);
    }

    virtual void operator()(const unsigned int level, VectorType &dst, const VectorType &src) const override {
      (void) level;
      if (temp_src_double.size() == 0) {
        temp_src_double.reinit(src, true);
        temp_dst_double.reinit(dst, true);
      }

      temp_src_double = src;
      temp_dst_double = dst;
      direct_solver.vmult(temp_dst_double, temp_src_double);
      dst = temp_dst_double;
    }

  private:
    dealii::TrilinosWrappers::SolverDirect direct_solver;
    // This solver only supports doubles
    mutable dealii::LinearAlgebra::distributed::Vector<double> temp_src_double;
    mutable dealii::LinearAlgebra::distributed::Vector<double> temp_dst_double;
  };

}