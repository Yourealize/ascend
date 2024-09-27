
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DepthToSpaceTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreBlockNum);   //每个小核需要处理的单位数据数
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreBlockNum);     //每个大核需要处理的单位数据数
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);         //每个单位数据中的数据数
  // TILING_DATA_FIELD_DEF(uint32_t, tileBlockNum);        //
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);        //大核数
  TILING_DATA_FIELD_DEF(uint32_t, dim1);
  TILING_DATA_FIELD_DEF(uint32_t, dim2);
  TILING_DATA_FIELD_DEF(uint32_t, dim3);
  TILING_DATA_FIELD_DEF(uint32_t, dim4);
  TILING_DATA_FIELD_DEF(uint32_t, dim5);
  TILING_DATA_FIELD_DEF(uint32_t, Batch);
  TILING_DATA_FIELD_DEF(uint32_t, DoubleBuffer);
  TILING_DATA_FIELD_DEF(uint32_t, Type);
  TILING_DATA_FIELD_DEF(uint32_t, Pad);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DepthToSpace, DepthToSpaceTilingData)
}
