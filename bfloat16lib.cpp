#include <torch/extension.h>

#include <ATen/CPUFloatType.h>
#include <ATen/Type.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/detail/Bfloat16HooksInterface.h>

#include "ATen/Allocator.h"
#include "ATen/CPUGenerator.h"
#include "ATen/DeviceGuard.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Utils.h"
#include "ATen/WrapDimUtils.h"
#include "c10/Bfloat16.h"
#include "ATen/core/TensorImpl.h"
#include "ATen/core/UndefinedTensorImpl.h"
#include "c10/util/Optional.h"
#include "TH/THGeneral.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "ATen/Config.h"
#include "ATen/CPUApplyUtils.h"

namespace at {

struct CPUBFloat16Type : public at::CPUTypeDefault {
  CPUBFloat16Type()
      : CPUTypeDefault(
            CPUTensorId(),
            /*is_variable=*/false,
            /*is_undefined=*/false) {}

  ScalarType scalarType() const override;
  caffe2::TypeMeta typeMeta() const override;
  Backend backend() const override;
  const char* toString() const override;
  size_t elementSizeInBytes() const override;
  TypeID ID() const override;

Tensor empty(IntList size, const TensorOptions & options) const override {
    return at::native::empty_cpu(/* actuals */ size, options);
}
Scalar _local_scalar_dense(const Tensor & self) const override {
    return Scalar(*self.data<at::BFloat16>());
}
Tensor & resize_(Tensor & self, IntList size) const override {
    return at::native::resize_cpu_(/* actuals */ self, size);
}
Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const override {
    if (src.type().scalarType() != at::ScalarType::Float)
        THError("s_copy_() for BFloat16 implemented only for float");

    using self_T = at::BFloat16;
    using src_T = float;
    at::CPU_tensor_apply2<self_T, src_T>(
      self, src, [](self_T& self_val, const src_T& src_val) {
        self_val = src_val;
      });
    return self;
}
Tensor mm(const Tensor & self, const Tensor & mat2) const override {
    if (self.is_sparse())
        THError("Operation on sparse tensor type not supported");

    // Copied from CPUFloatType::_th_mm()
    auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
                       CPUTensorId(), caffe2::TypeMeta::Make<at::BFloat16>(),
                       allocator(), false).release();
    auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
    result.resize_({ self.size(0), mat2.size(1) });
    auto self_ = checked_tensor_unwrap(self,"self",1, false, Backend::CPU, ScalarType::BFloat16);
    auto mat2_ = checked_tensor_unwrap(mat2,"mat2",2, false, Backend::CPU, ScalarType::BFloat16);

    if (self_->dim() != 2 || mat2_->dim() != 2)
        THError("matrices expected, got %dD, %dD tensors", self_->dim(), mat2_->dim());

    if (self_->size(1) != mat2_->size(0))
        THError("size mismatch");

    // Throw error if non-contiguous matrices
    if (self_->stride(1) != 1 || mat2_->stride(1) != 1)
        THError("mm() currently supported for row-major contiguous matrices only");

    // Multiply matrices of shapes (M, N) and (N, P)
    auto M = self_->size(0);
    auto N = self_->size(1);
    auto P = mat2_->size(1);

    // Get pointers to the tensor data elements
    auto m1_p = self_->data<at::BFloat16>();
    auto m2_p = mat2_->data<at::BFloat16>();
    auto r_p = result_->data<at::BFloat16>();

    // Standard O(n**3) matrix multiplication
    for (auto i = 0; i < M; ++i) {
        for (auto j = 0; j < P; ++j) {
            // Accumulate sum in float
            float sum = 0;
            for (auto k = 0; k < N; ++k) {
                sum += static_cast<float>(m1_p[i*N + k]) * static_cast<float>(m2_p[k*P + j]);
            }
            // Quantize sum to BFloat16
            r_p[i*P + j] = sum;
        }
    }

    result_->maybe_zero_dim(self_->dim() == 0 && mat2_->dim() == 0);
    return result;
}

};

struct BFloat16Hooks : public at::BFloat16HooksInterface {
  BFloat16Hooks(BFloat16HooksArgs) {}
  void registerBFloat16Type(Context* context) const override {
    context->registerType(
        Backend::CPU, ScalarType::BFloat16, new CPUBFloat16Type());
  }
};

ScalarType CPUBFloat16Type::scalarType() const {
  return ScalarType::BFloat16;
}

caffe2::TypeMeta CPUBFloat16Type::typeMeta() const {
  return scalarTypeToTypeMeta(ScalarType::BFloat16);
}

Backend CPUBFloat16Type::backend() const {
  return Backend::CPU;
}

const char* CPUBFloat16Type::toString() const {
  return "CPUBFloat16Type";
}

TypeID CPUBFloat16Type::ID() const {
  return TypeID::CPUBFloat16;
}

size_t CPUBFloat16Type::elementSizeInBytes() const {
  return sizeof(at::BFloat16);
}

REGISTER_BFLOAT16_HOOKS(BFloat16Hooks);

} // namespace at

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { }
