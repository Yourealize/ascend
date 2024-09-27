#!/bin/bash
echo "[Ascend910B1] Generating GroupNormV2_499e827d5d577e31090ae81a6c53b542 ..."
opc $1 --main_func=group_norm_v2 --input_param=/home/ma-user/work/GroupNormV2/build_out/op_kernel/binary/ascend910b/gen/GroupNormV2_499e827d5d577e31090ae81a6c53b542_param.json --soc_version=Ascend910B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/GroupNormV2_499e827d5d577e31090ae81a6c53b542.json ; then
  echo "$2/GroupNormV2_499e827d5d577e31090ae81a6c53b542.json not generated!"
  exit 1
fi

if ! test -f $2/GroupNormV2_499e827d5d577e31090ae81a6c53b542.o ; then
  echo "$2/GroupNormV2_499e827d5d577e31090ae81a6c53b542.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating GroupNormV2_499e827d5d577e31090ae81a6c53b542 Done"
