
#include "depth_to_space_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <vector>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

      DepthToSpaceTilingData tiling;
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    std::cout << "ubSize" << ubSize << std::endl;
    const gert::StorageShape* shape = context->GetInputShape(0);
    int64_t Block_size = *(context->GetAttrs()->GetInt(0));
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    std::cout << "typeLength: " << typeLength << std::endl;
    std::vector<uint64_t> dim(4, 1);
    int n=4;
    for (int j = shape->GetStorageShape().GetDimNum() - 1; j >= 0; --j) {
        dim[--n] = shape->GetStorageShape().GetDim(j);
    }
    const char *mode = context->GetAttrs()->GetAttrPointer<char>(1);
    const char *data_format = context->GetAttrs()->GetAttrPointer<char>(2);
    uint32_t smallCoreBlockNum=0,bigCoreBlockNum=0;   //每个小/大核需要处理的单位数据数
    uint32_t tileDataNum;    //每个单位数据中的数据数
    uint32_t tailBlockNum;   //大核数
    uint32_t DoubleBuffer;   //是都使用doublebuffer
    uint32_t Batch,dim1,dim2,dim3,dim4,dim5;
    uint32_t Type,Pad;
    if (strcmp(mode,"DCR") == 0 && strcmp(data_format,"NCHW") == 0){
        
        Batch = dim[0];
        dim4 = dim[2];
        dim5 = dim[3];
        dim3 = dim[1] / (Block_size*Block_size);
        dim1=Block_size,dim2 = Block_size;
        std::cout << "dim1: " << dim[1] << std::endl;
        std::cout << "dim2: " << dim[2] << std::endl;
        std::cout << "dim3: " << dim[3] << std::endl;
        Type = 0;
        coreNum = coreNum>(Batch)? Batch : coreNum;
        tailBlockNum = Batch % coreNum;
        tileDataNum = dim1*dim2*dim3*dim4*dim5;
        if((tileDataNum*typeLength)%64==0){
            std::cout << "Process0" << std::endl;
            Type=0;
            Pad=0;
        }else if ((tileDataNum*typeLength)%32==0){
            std::cout << "Process1" << std::endl;
            Type=1;
            Pad=0;
        }else{
            std::cout << "Process1" << std::endl;
            Type=1;
            Pad = 1;
        }
        
        if (tailBlockNum > 0 &&tailBlockNum<coreNum ){

            bigCoreBlockNum = Batch/coreNum + 1;
            smallCoreBlockNum = Batch/coreNum;
        }else{
            bigCoreBlockNum = Batch/coreNum;
            smallCoreBlockNum = Batch/coreNum;
        }
        if (dim2*typeLength*4<ubSize){

            DoubleBuffer = 2;
        }else 
            DoubleBuffer = 1;
        

    }else if (strcmp(mode,"CRD")==0 && strcmp(data_format,"NCHW")==0){
        
        Batch = dim[0];
        dim1 = dim[1] / (Block_size*Block_size);
        dim2 = Block_size,dim3 = Block_size;
        dim4 = dim[2];
        dim5 = dim[3];

        coreNum = coreNum>(Batch*dim1)? Batch*dim1 : coreNum;
        tailBlockNum = (Batch*dim1) % coreNum;
        std::cout << "bigCoreNum" << tailBlockNum << std::endl;
        tileDataNum = dim2*dim3*dim4*dim5;
        // if ((tileDataNum*typeLength)%64==0){
        //     Type=3;
        //     Pad=0;
        // }
        if ((tileDataNum*typeLength)%32==0){
            Type=1;
            Pad=0;
        }else{
            Type=1;
            Pad = 1;
        }
        if (tailBlockNum > 0 &&tailBlockNum<coreNum){
 
            bigCoreBlockNum = (Batch*dim1)/coreNum + 1;
            smallCoreBlockNum = (Batch*dim1)/coreNum;
        }else{
            bigCoreBlockNum = (Batch*dim1)/coreNum;
            smallCoreBlockNum = (Batch*dim1)/coreNum;
        }
        if (tileDataNum*typeLength*4<ubSize){

            DoubleBuffer = 2;
        }else 
            DoubleBuffer = 1;
        std::cout << "Type: " << Type << std::endl;
    }else if (strcmp(mode,"DCR")==0 && strcmp(data_format,"NHWC")==0){
        // std::cout << "Process2" << std::endl;
        Batch = dim[0];
        dim1 = dim[1];
        dim2 = dim[2];
        dim3=Block_size,dim4 = Block_size;
        dim5 = dim[3] / (Block_size*Block_size);
        Type = 2;
        coreNum = coreNum>(Batch*dim1)? Batch*dim1 : coreNum;
        std::cout << "coreNum" << coreNum << std::endl;
        tailBlockNum = (Batch*dim1) % coreNum;
        std::cout << "bigCoreNum" << tailBlockNum << std::endl;
        tileDataNum = dim2*dim3*dim4*dim5;
        if ((tileDataNum*typeLength)%32==0)
            Pad=0;
        else
            Pad=1;
        if (tailBlockNum > 0 &&tailBlockNum<coreNum){

            bigCoreBlockNum = (Batch*dim1)/coreNum + 1;
            smallCoreBlockNum = (Batch*dim1)/coreNum;
        }else{
            bigCoreBlockNum = (Batch*dim1)/coreNum;
            smallCoreBlockNum = (Batch*dim1)/coreNum;
        }
        if (dim4*dim5*typeLength*2<ubSize){

            DoubleBuffer = 2;
        }else 
            DoubleBuffer = 1;
    }else{

        Batch = dim[0];
        dim1 = dim[1];
        dim2 = dim[2];
        dim3 = dim[3] / (Block_size*Block_size);
        dim4 = Block_size,dim5 = Block_size;
        Type = 1;
        coreNum = coreNum>(Batch*dim1)? Batch*dim1 : coreNum;
        tailBlockNum = (Batch*dim1) % coreNum;
        tileDataNum = dim2*dim3*dim4*dim5;
        std::cout << "tileDataNum*typeLength % 32 = " << (tileDataNum*typeLength)%32 << std::endl;
        if ((tileDataNum*typeLength)%32==0){
            Type=3;
            Pad=0;
        }else{
            Type=3;
            Pad = 1;
        }
        if (tailBlockNum > 0 &&tailBlockNum<coreNum){
            /* code */
            bigCoreBlockNum = (Batch*dim1)/coreNum + 1;
            smallCoreBlockNum = (Batch*dim1)/coreNum;
        }else{
            bigCoreBlockNum = (Batch*dim1)/coreNum;
            smallCoreBlockNum = (Batch*dim1)/coreNum;
        }
        if (tileDataNum*typeLength*4<ubSize){
 
            DoubleBuffer = 2;
        }else 
            DoubleBuffer = 1;
    }
    std::cout << "Type: " << Type << std::endl;
    std::cout << "dim2: " << dim2 << std::endl;
    tiling.set_smallCoreBlockNum(smallCoreBlockNum);
    tiling.set_bigCoreBlockNum(bigCoreBlockNum);
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_tailBlockNum(tailBlockNum);
    tiling.set_dim1(dim1);
    tiling.set_dim2(dim2);
    tiling.set_dim3(dim3);
    tiling.set_dim4(dim4);
    tiling.set_dim5(dim5);
    tiling.set_Batch(Batch);
    tiling.set_DoubleBuffer(DoubleBuffer);
    tiling.set_Type(Type);
    tiling.set_Pad(Pad);
    // if(Type!=2)
    context->SetBlockDim(coreNum);
    // else context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;

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
class DepthToSpace : public OpDef {
public:
    explicit DepthToSpace(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("Block_size").Int();
        this->Attr("mode").AttrType(OPTIONAL).String("DCR");
        this->Attr("Data_format").AttrType(OPTIONAL).String("NHWC");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(DepthToSpace);
}
