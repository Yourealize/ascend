#!/bin/bash
echo "[Ascend910B1] Generating DepthToSpace_18665e9759bd12a2adf7fab69edc49c3 ..."
opc $1 --main_func=depth_to_space --input_param=/home/ma-user/work/DepthToSpace/build_out/op_kernel/binary/ascend910b/gen/DepthToSpace_18665e9759bd12a2adf7fab69edc49c3_param.json --soc_version=Ascend910B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/DepthToSpace_18665e9759bd12a2adf7fab69edc49c3.json ; then
  echo "$2/DepthToSpace_18665e9759bd12a2adf7fab69edc49c3.json not generated!"
  exit 1
fi

if ! test -f $2/DepthToSpace_18665e9759bd12a2adf7fab69edc49c3.o ; then
  echo "$2/DepthToSpace_18665e9759bd12a2adf7fab69edc49c3.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating DepthToSpace_18665e9759bd12a2adf7fab69edc49c3 Done"
