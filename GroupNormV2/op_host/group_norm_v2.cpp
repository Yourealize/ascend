
#include "group_norm_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include<vector>
#include<math.h>


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  GroupNormV2TilingData tiling;
  uint64_t ubSize;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  auto coreNum = ascendcPlatform.GetCoreNum();
  std::cout << ubSize << std::endl;

  const gert::StorageShape* shape = context->GetInputShape(0);
  int num_groups = *context->GetAttrs()->GetInt(0);
  float eps = *context->GetAttrs()->GetFloat(2);
  uint64_t length = std::max<uint64_t>(length, context->GetInputShape(0)->GetStorageShape().GetDimNum());
  std::vector<uint64_t> dim(4, 1);
  int n=length;
  for (int j = shape->GetStorageShape().GetDimNum() - 1; j >= 0; --j) {
      dim[--n] = shape->GetStorageShape().GetDim(j);
  }
    
  uint32_t Batch = dim[0],Channel = dim[1];
  uint32_t totalSize = dim[0]*dim[1]*dim[2]*dim[3];
  uint32_t groupSize = dim[1]/num_groups*dim[2]*dim[3];
  uint32_t channelSize = dim[2]*dim[3];
  uint32_t sizeofdatatype;
  auto dt = context->GetInputDesc(0)->GetDataType();
  if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        sizeofdatatype = 2;
    }
    else {
        sizeofdatatype = 4;
    }
  uint64_t rest = ((ubSize-Batch*num_groups*sizeofdatatype) / 2 );
  uint64_t groupUBSize = rest/2;
  groupUBSize = (groupUBSize >> 8) << 8;
  uint32_t tileDataNum,tailDataNum,tileNum;
  // tileDataNum = min(groupUBSize / sizeofdatatype,;
  tileDataNum = (256/sizeofdatatype)*(256/sizeofdatatype);
  tailDataNum = groupSize % tileDataNum;
  if(tailDataNum>0) tileNum = groupSize / tileDataNum +1;
  else {
      tileNum = groupSize / tileDataNum;
      tailDataNum = tileDataNum;
  }
  uint32_t tileChannelData,tailChannelData,tileChannelNum;
    tileChannelData = 64>channelSize?channelSize:64;
    tailChannelData = channelSize%tileChannelData;
    if(tailChannelData > 0) tileChannelNum = channelSize/tileChannelData + 1;
    else{
        tileChannelNum = channelSize/tileChannelData;
        tailChannelData = tileChannelData;
    }
    coreNum = coreNum>(Batch*num_groups)?(Batch*num_groups):coreNum;
    uint32_t bigCoreBlockNum,smallCoreBlockNum;
    uint32_t tailBlockNum = (Batch*num_groups)%coreNum;
    if(tailBlockNum == coreNum || tailBlockNum==0){
        bigCoreBlockNum = Batch*num_groups/coreNum;
        smallCoreBlockNum = Batch*num_groups/coreNum;
    }else{
        bigCoreBlockNum = Batch*num_groups/coreNum+1;
        smallCoreBlockNum = Batch*num_groups/coreNum;
    }
    
  //   uint64_t channelUBSize = rest/2;
  //   channelBlockNum = (channelBlockNum >> 5);
//   coreNum = coreNum>(Batch*num_groups) ? coreNum : (Batch*num_groups);
//   uint32_t tailBlockNum = (Batch*num_groups) % coreNum;
//   uint32_t bigCoreBlockNum,smallCoreBlockNum;
//   if (tailBlockNum < coreNum){
//         bigCoreBlockNum = (Batch*num_groups)/coreNum + 1;
//         smallCoreBlockNum = (Batch*num_groups)/coreNum;
//     }else{
//         bigCoreBlockNum = (Batch*num_groups)/coreNum;
//         smallCoreBlockNum = (Batch*num_groups)/coreNum;
//     }
  if((groupSize*sizeofdatatype)%256==0) context->SetBlockDim(coreNum);
  else context->SetBlockDim(1);
  tiling.set_Batch(Batch);
  tiling.set_num_groups(num_groups);
  tiling.set_Channel(Channel);
  tiling.set_epsilon(eps);
  tiling.set_totalSize(totalSize);
  tiling.set_groupSize(groupSize);
  tiling.set_channelSize(channelSize);
  tiling.set_tileDataNum(tileDataNum);
  tiling.set_tailDataNum(tailDataNum);
  tiling.set_tileNum(tileNum);
  tiling.set_tileChannelData(tileChannelData);
  tiling.set_tailChannelData(tailChannelData);
  tiling.set_tileChannelNum(tileChannelNum);
  tiling.set_tailBlockNum(tailBlockNum);
  tiling.set_bigCoreBlockNum(bigCoreBlockNum);
  tiling.set_smallCoreBlockNum(smallCoreBlockNum);
//   tiling.set_tailBlockNum(tailBlockNum);
//   tiling.set_bigCoreBlockNum(bigCoreBlockNum);
//   tiling.set_smallCoreBlockNum(smallCoreBlockNum);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class GroupNormV2 : public OpDef {
public:
    explicit GroupNormV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_groups").Int();
        this->Attr("data_format").AttrType(OPTIONAL).String("NCHW");
        this->Attr("eps").AttrType(OPTIONAL).Float(0.0001);
        this->Attr("is_training").AttrType(OPTIONAL).Bool(true);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(GroupNormV2);
}
