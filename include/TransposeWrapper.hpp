#pragma once

namespace solver {

  template <typename OperatorType, typename VectorType>
  class TransposeWrapper {
  public:
    TransposeWrapper(const OperatorType &op) : op(op) {}
  
    void vmult(VectorType &dst, const VectorType &src) const {
      op.Tvmult(dst, src);
    }
  
  private:
    const OperatorType &op;
  
  };
  
}
