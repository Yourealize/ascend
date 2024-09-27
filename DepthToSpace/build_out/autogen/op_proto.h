#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(DepthToSpace)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .REQUIRED_ATTR(Block_size, Int)
    .ATTR(mode, String, "DCR")
    .ATTR(Data_format, String, "NHWC")
    .OP_END_FACTORY_REG(DepthToSpace);

}

#endif
