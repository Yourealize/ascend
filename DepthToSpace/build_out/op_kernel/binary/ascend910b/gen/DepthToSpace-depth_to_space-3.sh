#!/bin/bash
echo "[Ascend910B1] Generating DepthToSpace_5d7c211aab8faa96d43ebd3f7248387b ..."
opc $1 --main_func=depth_to_space --input_param=/home/ma-user/work/DepthToSpace/build_out/op_kernel/binary/ascend910b/gen/DepthToSpace_5d7c211aab8faa96d43ebd3f7248387b_param.json --soc_version=Ascend910B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/DepthToSpace_5d7c211aab8faa96d43ebd3f7248387b.json ; then
  echo "$2/DepthToSpace_5d7c211aab8faa96d43ebd3f7248387b.json not generated!"
  exit 1
fi

if ! test -f $2/DepthToSpace_5d7c211aab8faa96d43ebd3f7248387b.o ; then
  echo "$2/DepthToSpace_5d7c211aab8faa96d43ebd3f7248387b.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating DepthToSpace_5d7c211aab8faa96d43ebd3f7248387b Done"
