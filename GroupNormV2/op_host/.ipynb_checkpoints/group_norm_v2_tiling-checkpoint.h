#include "register/tilingdata_base.h"
#include<vector>
namespace optiling {
BEGIN_TILING_DATA_DEF(GroupNormV2TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, num_groups);
  TILING_DATA_FIELD_DEF(uint32_t, Batch);
  TILING_DATA_FIELD_DEF(uint32_t, Channel);
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreBlockNum);   //每个小核需要处理的单位数据数
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreBlockNum);     //每个大核需要处理的单位数据数
  TILING_DATA_FIELD_DEF(uint32_t, totalSize);
  TILING_DATA_FIELD_DEF(uint32_t, groupSize);
  TILING_DATA_FIELD_DEF(uint32_t, channelSize);
  TILING_DATA_FIELD_DEF(float, epsilon);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileChannelData);
  TILING_DATA_FIELD_DEF(uint32_t, tailChannelData);
  TILING_DATA_FIELD_DEF(uint32_t, tileChannelNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormV2, GroupNormV2TilingData)
}
