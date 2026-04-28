#pragma once

#include <memory>


namespace solver {
  template <typename BlockVectorType, typename InnerSolverZero, typename InnerSolverSecond, typename CouplingOperator>
  class BlockGSPreconditioner {
  public:
    using VectorType = typename BlockVectorType::BlockType;
  
    BlockGSPreconditioner(std::shared_ptr<const InnerSolverZero> inv_zero_op, std::shared_ptr<const InnerSolverSecond> inv_second_op, std::shared_ptr<const CouplingOperator> coupling_op) :
      op_invA00(inv_zero_op),
      op_invA22(inv_second_op),
      op_A20(coupling_op)
    {}

    void clear();
    void initialize(const BlockVectorType &);
    void vmult(BlockVectorType &, const BlockVectorType &) const;
    void Tvmult(BlockVectorType &, const BlockVectorType &) const;


  private:
    std::shared_ptr<const InnerSolverZero> op_invA00;
    std::shared_ptr<const InnerSolverSecond> op_invA22;
    std::shared_ptr<const CouplingOperator> op_A20;

    mutable VectorType tmp_A20_dst0;
    mutable VectorType tmp_rhs_2;

    mutable VectorType tmp_A20T_dst2;
    mutable VectorType tmp_rhs_0;

  };

}