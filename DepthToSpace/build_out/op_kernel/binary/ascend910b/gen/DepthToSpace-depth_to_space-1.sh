#!/bin/bash
echo "[Ascend910B1] Generating DepthToSpace_715b3263d6a78f0490d06b59e95b2eda ..."
opc $1 --main_func=depth_to_space --input_param=/home/ma-user/work/DepthToSpace/build_out/op_kernel/binary/ascend910b/gen/DepthToSpace_715b3263d6a78f0490d06b59e95b2eda_param.json --soc_version=Ascend910B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/DepthToSpace_715b3263d6a78f0490d06b59e95b2eda.json ; then
  echo "$2/DepthToSpace_715b3263d6a78f0490d06b59e95b2eda.json not generated!"
  exit 1
fi

if ! test -f $2/DepthToSpace_715b3263d6a78f0490d06b59e95b2eda.o ; then
  echo "$2/DepthToSpace_715b3263d6a78f0490d06b59e95b2eda.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating DepthToSpace_715b3263d6a78f0490d06b59e95b2eda Done"
