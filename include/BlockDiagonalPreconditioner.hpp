#pragma once

#include <memory>

namespace solver {

  template <typename BlockVectorType, typename InnerPreconditionerZero, typename InnerPreconditionerSecond>
  class BlockDiagonalPreconditioner {
  
  public:
    BlockDiagonalPreconditioner(std::shared_ptr<const InnerPreconditionerZero> prec_zero, std::shared_ptr<const InnerPreconditionerSecond> prec_second) :
      prec_zero(prec_zero),
      prec_second(prec_second)
    {}

    void vmult(BlockVectorType &, const BlockVectorType &) const;
    void Tvmult(BlockVectorType &, const BlockVectorType &) const;

  private:
    std::shared_ptr<const InnerPreconditionerZero> prec_zero;
    std::shared_ptr<const InnerPreconditionerSecond> prec_second;

  };

}
