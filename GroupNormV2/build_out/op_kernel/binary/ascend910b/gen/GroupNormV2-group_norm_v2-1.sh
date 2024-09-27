#!/bin/bash
echo "[Ascend910B1] Generating GroupNormV2_783e55641b4dfa21360521bf61bcee00 ..."
opc $1 --main_func=group_norm_v2 --input_param=/home/ma-user/work/GroupNormV2/build_out/op_kernel/binary/ascend910b/gen/GroupNormV2_783e55641b4dfa21360521bf61bcee00_param.json --soc_version=Ascend910B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/GroupNormV2_783e55641b4dfa21360521bf61bcee00.json ; then
  echo "$2/GroupNormV2_783e55641b4dfa21360521bf61bcee00.json not generated!"
  exit 1
fi

if ! test -f $2/GroupNormV2_783e55641b4dfa21360521bf61bcee00.o ; then
  echo "$2/GroupNormV2_783e55641b4dfa21360521bf61bcee00.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating GroupNormV2_783e55641b4dfa21360521bf61bcee00 Done"
