#pragma once

#include <memory>

namespace solver {
  template <typename BlockVectorType, typename InnerPreconditionerZero, typename InnerPreconditionerSecond>
  class BlockDiagonalPreconditioner {
  public:
    BlockDiagonalPreconditioner(std::shared_ptr<const InnerPreconditionerZero> prec_zero, 
                                std::shared_ptr<const InnerPreconditionerSecond> prec_second) :
      prec_zero(prec_zero),
      prec_second(prec_second)
    {}

    void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
      prec_zero->vmult(dst.block(0), src.block(0));
      prec_second->vmult(dst.block(1), src.block(1));
    }

    void Tvmult(BlockVectorType &dst, const BlockVectorType &src) const {
      prec_zero->Tvmult(dst.block(0), src.block(0));
      prec_second->Tvmult(dst.block(1), src.block(1));
    }

  private:
    std::shared_ptr<const InnerPreconditionerZero> prec_zero;
    std::shared_ptr<const InnerPreconditionerSecond> prec_second;
  };
}
