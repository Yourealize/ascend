#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GroupNormV2)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(gamma, ge::TensorType::ALL())
    .INPUT(beta, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(mean, ge::TensorType::ALL())
    .OUTPUT(rstd, ge::TensorType::ALL())
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(eps, Float, 0.0001)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(GroupNormV2);

}

#endif
