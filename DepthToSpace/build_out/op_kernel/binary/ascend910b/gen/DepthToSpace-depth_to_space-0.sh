#!/bin/bash
echo "[Ascend910B1] Generating DepthToSpace_3642b003a2cc75afd913ddad7bc88feb ..."
opc $1 --main_func=depth_to_space --input_param=/home/ma-user/work/DepthToSpace/build_out/op_kernel/binary/ascend910b/gen/DepthToSpace_3642b003a2cc75afd913ddad7bc88feb_param.json --soc_version=Ascend910B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/DepthToSpace_3642b003a2cc75afd913ddad7bc88feb.json ; then
  echo "$2/DepthToSpace_3642b003a2cc75afd913ddad7bc88feb.json not generated!"
  exit 1
fi

if ! test -f $2/DepthToSpace_3642b003a2cc75afd913ddad7bc88feb.o ; then
  echo "$2/DepthToSpace_3642b003a2cc75afd913ddad7bc88feb.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating DepthToSpace_3642b003a2cc75afd913ddad7bc88feb Done"
