#include "kernel_operator.h"
using namespace AscendC;

template<typename TYPE_X, typename TYPE_Y> class KernelDepthToSpace{
    public:
        __aicore__ inline KernelDepthToSpace() {}
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,uint32_t smallCoreBlockNum,
                                    uint32_t bigCoreBlockNum,uint32_t tileDataNum,
                                    uint32_t tailBlockNum,uint32_t dim1,uint32_t dim2,
                                    uint32_t dim3,uint32_t dim4,uint32_t dim5,uint32_t Pad,
                                    uint32_t Batch,uint32_t DoubleBuffer,uint32_t Type) {
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            // printf("Init\n");
            
            uint32_t coreNum = GetBlockIdx();
            uint32_t globalBufferIndex;
            this->tileDataNum = tileDataNum;
            this->Pad = Pad;
            this->dim1 = dim1;
            this->dim2 = dim2;
            this->dim3 = dim3;
            this->dim4 = dim4;
            this->dim5 = dim5;
            this->Batch = Batch;
            
            if (coreNum < tailBlockNum) {
                this->CoreBlockNum = bigCoreBlockNum;
                this->CoreDataNum = bigCoreBlockNum*tileDataNum;
                globalBufferIndex = bigCoreBlockNum*tileDataNum * GetBlockIdx();
            }else{
                this->CoreBlockNum = smallCoreBlockNum;
                this->CoreDataNum = smallCoreBlockNum*tileDataNum;
                globalBufferIndex = tailBlockNum *bigCoreBlockNum*tileDataNum 
                                    + (GetBlockIdx() - tailBlockNum)*tileDataNum*smallCoreBlockNum;
            }
            // printf("CoreBlocNum= %d\n",this->CoreBlockNum);
            // if(coreNum==1) printf("globalBuffer = %d\n",globalBufferIndex);
            // if(Type!=2){
            xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->CoreDataNum);
            yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->CoreDataNum);
            // else{
               // xGm.SetGlobalBuffer((__gm__ TYPE_X*)x , Batch*dim1*this->tileDataNum);
            // yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y , Batch*dim1*this->tileDataNum); 
            // }
            if (Type == 1){
                pipe.InitBuffer(inQueuedX, DoubleBuffer, this->dim5*this->dim3 * sizeof(TYPE_X));
            }else if(Type==2){
                pipe.InitBuffer(inQueuedX, DoubleBuffer, this->dim5*this->dim4* sizeof(TYPE_X));
                // pipe.InitBuffer(outQueueY, DoubleBuffer, this->tileDataNum * sizeof(TYPE_Y));
            }
            else if(Type==0){
                pipe.InitBuffer(inQueuedX, DoubleBuffer, this->dim5*this->dim2 * sizeof(TYPE_X));
            }else{
                pipe.InitBuffer(inQueueX, DoubleBuffer, this->tileDataNum * sizeof(TYPE_X));
                pipe.InitBuffer(outQueueY, DoubleBuffer, this->tileDataNum * sizeof(TYPE_Y));
            }
    }
                

        __aicore__ inline void Process0(){
            // std::cout << "Process0" << std::endl;
            printf("%d\n",this->tileDataNum);
            printf("%d\n",this->dim1*this->dim2*this->dim3*this->dim4*this->dim5);
            uint32_t coreNum = GetBlockIdx();
            // if(coreNum==0){
            //     for(int i=0;i<this->tileDataNum;i++)
            //         printf("%d ",xGm.GetValue(i));
            // }
            if(this->tileDataNum*sizeof(TYPE_X)%64==0)
            for (uint32_t b = 0; b < this->CoreBlockNum; b++){
                for (uint32_t b1 = 0; b1 < this->dim1; b1++){
                    for (uint32_t b2 = 0; b2 < this->dim2; b2++){
                        for(uint32_t cb=0;cb<this->dim3;cb++){
                            for(uint32_t h=0;h<this->dim4;h++){
                                for(uint32_t w = 0;w<this->dim5;w++)
                                {
                                    auto input_index = b*this->tileDataNum +b1*this->dim2*this->dim3*this->dim4*this->dim5
                                                    +b2*this->dim3*this->dim4*this->dim5+cb*this->dim4*this->dim5
                                                    + h*this->dim5+w;
                                    // auto output_index = (this->dim2*input_index)%this->tileDataNum;
                                    auto output_index = b*this->tileDataNum + cb*this->dim4*this->dim1*this->dim5*this->dim2
                                                        +h*this->dim1*this->dim5*this->dim2+b1*this->dim5*this->dim2
                                                        +w*this->dim2+b2;
                                    output_index = output_index % this->tileDataNum;
                                    // auto xg=xGm.GetValue(input_index);
                                    yGm.SetValue(output_index,xGm.GetValue(input_index));
                                    // auto yg = yGm.GetValue(output_index);
                                    if(coreNum==0)
                                        printf("%d ",input_index);
                                }
                            }
                        }
                    }
                }
                
            }
            else{
                 for (uint32_t b = 0; b < this->CoreBlockNum; b++)
                     for(uint32_t cb=0;cb<this->dim3;cb++)
                         for(uint32_t h=0;h<this->dim4;h++)
                             for(uint32_t b1=0;b1<this->dim1;b1++){
                                  LocalTensor<TYPE_X> xLocal = inQueuedX.AllocTensor<TYPE_X>();
                                 // if(coreNum==3&&b1>=0&&b1<=3&&h==0){
                                 //     printf("\n**********************\n");
                                 // printf("\nGM: \n");
                                 // }
                                 for(uint32_t w = 0;w<this->dim5;w++)
                                     for(uint32_t b2 =0 ;b2<this->dim2;b2++){
                                         auto input_index = b*this->tileDataNum +b1*this->dim2*this->dim3*this->dim4*this->dim5
                                                    +b2*this->dim3*this->dim4*this->dim5+cb*this->dim4*this->dim5
                                                    + h*this->dim5+w;
                                    // auto output_index = (this->dim2*input_index)%this->tileDataNum;
                                        // auto output_index = b*this->tileDataNum + cb*this->dim4*this->dim1*this->dim5*this->dim2
                                        //                 +h*this->dim1*this->dim5*this->dim2+b1*this->dim5*this->dim2
                                        //                 +w*this->dim2+b2;
                                         auto output_index = w*this->dim2+b2;
                                         xLocal.SetValue(output_index,xGm.GetValue(input_index));
//                                          if(coreNum==0&&b1>=0&&b1<=3&&h==0){
                                             
//                                          printf("%d ",xGm.GetValue(input_index));}
                                     }
                                 if((this->dim5*this->dim2*sizeof(TYPE_X))%32!=0){
                                     DataCopyExtParams copyParams;
                                    copyParams.blockCount = 1;
                                    copyParams.blockLen = this->dim5*this->dim2 * sizeof(TYPE_X);
                                    copyParams.srcStride = 0;
                                    copyParams.dstStride = 0;
                                     DataCopyPad(yGm[b*this->tileDataNum+cb*this->dim4*this->dim1*this->dim5*this->dim2
                                                        +h*this->dim1*this->dim5*this->dim2+b1*this->dim5*this->dim2], xLocal,copyParams );
                                 }else{
                                     
                                 DataCopy(yGm[b*this->tileDataNum+cb*this->dim4*this->dim1*this->dim5*this->dim2
                                                        +h*this->dim1*this->dim5*this->dim2+b1*this->dim5*this->dim2], xLocal, this->dim5*this->dim2);
                                 }
                                 inQueuedX.FreeTensor(xLocal); 
                                 
                             }
                    // inQueuedX.FreeTensor(xLocal); 
            }
            
            
        }
    // __aicore__ inline void Process0(){
    //     // uint32_t coreNum = GetBlockIdx();
    //     // LocalTensor<TYPE_X> xLocal = inQueuedX.AllocTensor<TYPE_X>();
    //     // if(coreNum==0)
    //     // for(int i=0;i<this->tileDataNum;i++)
    //     //     printf("%d ",xGm.GetValue(i));
    //     // // printf("\n\n*********************************************************************\n\n");
    //     // // DataCopyExtParams copyParams;
    //     // //         copyParams.blockCount = 8;
    //     // //         copyParams.blockLen =  sizeof(TYPE_X);
    //     // //         copyParams.srcStride = this->dim3*dim4*dim5*sizeof(TYPE_X)-4;
    //     // //         copyParams.dstStride = 0;
    //     // // DataCopyPadExtParams<TYPE_X> padParams{true, 0, 0, 0};
    //     // // DataCopyPad(xLocal, xGm, copyParams, padParams);
    //     // uint32_t xg = xLocal.GetSize();
    //     // if(coreNum==0)
    //     // for(int i=0;i<xg;i++)
    //     //     printf("%d ",xLocal.GetValue(i));
    //     // printf("\n");
    //     uint32_t loopCount = this->CoreBlockNum;
    //     uint32_t j=0;
    //     for(uint32_t i=0;i<loopCount;i++){
    //        for(uint32_t x=0;x<this->dim3*this->dim4*this->dim5;x++)
    //            for(uint32_t b2=0;b2<this->dim2;b2++){
    //                auto input_index = i*this->tileDataNum+b2*this->dim3*this->dim4*this->dim5+x;
    //                yGm.SetValue(j,xGm.GetValue(input_index));
    //                j++;
    //            }
    //     }
    // }

        __aicore__ inline void Process1(){

            printf("Process1");
            for(uint32_t b = 0; b < this->CoreBlockNum; b++)
                for(uint32_t d4=0;d4<this->dim4;d4++)
                    for(uint32_t d2=0;d2<this->dim2;d2++){
                    LocalTensor<TYPE_X> xLocal = inQueuedX.AllocTensor<TYPE_X>();
                        for(uint32_t d5=0;d5<this->dim5;d5++)
                            for(uint32_t d3=0;d3<this->dim3;d3++){
                                auto input_index = b*this->tileDataNum+d2*this->dim3*this->dim4*this->dim5
                                            +d3*this->dim4*this->dim5+d4*this->dim5+d5;
                                auto output_index = d5*this->dim3+d3;
                                xLocal.SetValue(output_index,xGm.GetValue(input_index));
                            }
                        if((this->dim5*this->dim2*sizeof(TYPE_X))%32!=0){
                                     DataCopyExtParams copyParams;
                                    copyParams.blockCount = 1;
                                    copyParams.blockLen = this->dim5*this->dim2 * sizeof(TYPE_X);
                                    copyParams.srcStride = 0;
                                    copyParams.dstStride = 0;
                                     DataCopyPad(yGm[b*this->tileDataNum+d4*this->dim2*this->dim5*this->dim3+d2*this->dim5*this->dim3],xLocal,copyParams );
                                 }else{
                                     
                                 DataCopy(yGm[b*this->tileDataNum+d4*this->dim2*this->dim5*this->dim3+d2*this->dim5*this->dim3], xLocal, this->dim5*this->dim2);
                                 }
                                 inQueuedX.FreeTensor(xLocal); 
                        
                    }
        }

        __aicore__ inline void Process2(){
            // printf("Process2\n");
            uint32_t coreNum = GetBlockIdx();
            // std::cout << "Process2" << std::endl;
            uint32_t loopCount = this->CoreBlockNum;
            // printf("%d %d %d\n",loopCount,this->dim2,this->dim3);
//             if((this->dim4*this->dim5*sizeof(TYPE_X))%32==0)
//             for(uint32_t b=0;b<this->Batch;b++)
//             for (uint32_t d1 = 0; d1 < this->dim1; d1++)
//                 for (uint32_t w = 0; w < this->dim2; w++)
//                     for(uint32_t bs=0;bs<this->dim3;bs++)
//                         for(uint32_t bs2=0;bs2<this->dim4;bs2++)
//                         for(uint32_t bc=0;bc<this->dim5;bc++){
//                         // LocalTensor<TYPE_X> xLocal = inQueuedX.AllocTensor<TYPE_X>();
//                         auto input_index = b*this->dim1*this->dim2*this->dim3*this->dim4*this->dim5
//                                             +d1*this->dim2*this->dim3*this->dim4*this->dim5+w*this->dim3*this->dim4*this->dim5
//                                             +bs*this->dim4*this->dim5+bs2*this->dim5+bc;
//                         auto output_index =  b*this->dim1*this->dim2*this->dim3*this->dim4*this->dim5
//                                             +d1*this->dim2*this->dim3*this->dim4*this->dim5+bs*this->dim2*this->dim4*this->dim5
//                                             +w*this->dim4*this->dim5+bs2*this->dim5+bc;
                        
//                         // if(coreNum==10) printf("%d %d**",output_index,input_index);
//                             yGm.SetValue(output_index,xGm.GetValue(input_index));
                            
                        // if((this->dim4*this->dim5*sizeof(TYPE_X))%32==0)
                        // DataCopy(xLocal,xGm[input_index],this->dim4*this->dim5);
                        // else{
                        //     DataCopyExtParams copyParams;
                        //             copyParams.blockCount = 1;
                        //             copyParams.blockLen = this->dim5*this->dim4 * sizeof(TYPE_X);
                        //             copyParams.srcStride = 0;
                        //             copyParams.dstStride = 0;
                        //     DataCopyPadExtParams<TYPE_X> padParams{true, 0, 0, 0};
                        //              DataCopyPad(xLocal, xGm[input_index],copyParams ,padParams);
                        // }
                        // if(coreNum==0) 
                        //     for(int i=0;i<this->dim4*this->dim5;i++)
                        //         printf("%f ",xLocal.GetValue(i));
                        // if((this->dim4*this->dim5*sizeof(TYPE_X))%32==0)
                        //     DataCopy(yGm[i*this->tileDataNum],xLocal,this->tileDataNum);
                        // else{
                        //     DataCopyExtParams copyParams;
                        //             copyParams.blockCount = 1;
                        //             copyParams.blockLen = this->tileDataNum * sizeof(TYPE_X);
                        //             copyParams.srcStride = 0;
                        //             copyParams.dstStride = 0;
                        //     DataCopyPad(yGm[i*this->tileDataNum],xLocal,copyParams);
                        // }
                        // inQueuedX.FreeTensor(xLocal); 
                        // CopyIn2(i,w,bs);
                        // CopyOut2(i,w,bs);
                    // }
            // else{
                // uint32_t loopCount = this->CoreBlockNum;
            for(uint32_t b=0;b<loopCount;b++)
                for(uint32_t b1=0 ;b1<this->dim3;b1++)
                    for(uint32_t w = 0;w<this->dim2;w++){
                        CopyIn2(b,b1,w);
                    }
//                 for(uint32_t b=0;b<loopCount;b++)
//                     for(uint32_t b1=0 ;b1<this->dim3;b1++)
                        
//                         for(uint32_t w = 0;w<this->dim2;w++){
//                             LocalTensor<TYPE_X> xLocal = inQueuedX.AllocTensor<TYPE_X>();
//                             event_t eventIDMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
//                             // for(uint32_t b2 = 0;b2<this->dim4;b2++)
//                             //     for(uint32_t bc=0;bc<this->dim5;bc++){
//                             //         auto input_index = b*this->tileDataNum+w*this->dim3*this->dim4*this->dim5
//                             //                             +b1*this->dim4*this->dim5+b2*this->dim5+bc;
//                             //         auto output_index =b2*this->dim5+bc;
//                             //         xLocal.SetValue(output_index,xGm.GetValue(input_index));
//                             //     }
//                             auto in = b*this->tileDataNum+w*this->dim3*this->dim4*this->dim5+b1*this->dim4*this->dim5;
//                              if((this->dim4*this->dim5*sizeof(TYPE_X))%32==0){
//                                 DataCopy(xLocal,xGm[in],this->dim4*this->dim5);
//                                 SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
//                              }
//                             else{
//                                 DataCopyExtParams copyParams;
//                                     copyParams.blockCount = 1;
//                                     copyParams.blockLen = this->dim5*this->dim4 * sizeof(TYPE_X);
//                                     copyParams.srcStride = 0;
//                                     copyParams.dstStride = 0;
//                                 DataCopyPadExtParams<TYPE_X> padParams{true, 0, 0, 0};
//                                      DataCopyPad(xLocal, xGm[in],copyParams ,padParams);
//                                 SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
//                             }
//                             auto out = b*this->tileDataNum+b1*this->dim2*this->dim5*this->dim4+w*this->dim4*this->dim5;
//                             // if(coreNum==0&&b1==0&&b==0)
//                             //     for(int i=0;i<xLocal.GetSize();i++)
//                             //         printf("%f ",xLocal.GetValue(i));

//                                 // printf("%d ",out);
//                         // for(int i=0;i<dim4*dim5;i++)
//                         //     printf("%f ",xLocal.GetValue(i));
//                         if((this->dim5*this->dim4*sizeof(TYPE_X))%32!=0){
//                             DataCopyExtParams copyParams;
//                             copyParams.blockCount = 1;
//                             copyParams.blockLen = this->dim5*this->dim4* sizeof(TYPE_X);
//                             copyParams.srcStride = 0;
//                             copyParams.dstStride = 0;
//                             WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
//                             DataCopyPad(yGm[out],xLocal,copyParams );
//                         }else{
//                             WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);         
//                             DataCopy(yGm[out], xLocal, this->dim5*this->dim4);
//                         }
//                         inQueuedX.FreeTensor(xLocal); 
//                     }
                                    
            // }
                
        }
    
    __aicore__ inline void Process3(){
            // std::cout << "Process1" << std::endl;
        printf("Process3");
            uint32_t loopCount = this->CoreBlockNum;
            printf("loopCount = %d\n",loopCount);
            for (uint32_t i = 0; i < loopCount; i++)
            {
                CopyIn1(i);
                CopyOut1(i);
            }
    }
    
        

    private:
        __aicore__ inline void CopyIn1(uint32_t progress){
            LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
            uint32_t coreNum = GetBlockIdx();
            if(coreNum==0) {
                for (int i=0;i<60;i++)
                    printf("%d ",xGm.GetValue(progress * this->tileDataNum+i));
            }
            if (this->Pad==0)
            {
                DataCopy(xLocal, xGm[progress * this->tileDataNum], this->tileDataNum*sizeof(TYPE_X));
            }else
            {
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = this->tileDataNum * sizeof(TYPE_X);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPadExtParams<TYPE_X> padParams{true, 0, 0, 0};
                 DataCopyPad(xLocal, xGm[progress * this->tileDataNum], copyParams, padParams);
            }
            auto xl = xLocal.GetSize();
            printf("tileDataNum = %d\n",this->tileDataNum);
            printf("xLobal size = %d\n",xl);
            if(coreNum==0)
            for(int i=0;i<xl;i++)
                printf("%d ",xLocal.GetValue(i));
            
            
            
            inQueueX.EnQue(xLocal);
        }

        __aicore__ inline void CopyOut1(int32_t progress){
            LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
            LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
            uint32_t coreNum = GetBlockIdx();
            
            for (int32_t d2 = 0; d2 < this->dim2; d2++)
                for (int32_t d3 = 0; d3 < this->dim3; d3++)
                    for (int32_t d4 = 0; d4 < this->dim4; d4++)
                        for(int32_t d5 = 0;d5 < this->dim5;d5++){
                            auto input_index = d2*this->dim3*this->dim4*this->dim5+d3*this->dim4*this->dim5
                                                +d4*this->dim5 + d5;
                            auto output_index = d4*this->dim2*this->dim5*this->dim3+d2*this->dim5*this->dim3
                                                +d5*this->dim3+d3;
                            yLocal.SetValue(output_index,xLocal.GetValue(input_index));
                            if(coreNum==0){
                                auto xg = xLocal.GetValue(input_index);
                                printf("%d ",xg);
                            }
                        }
            if(this->Pad==0)
                DataCopy(yGm[progress * this->tileDataNum], yLocal, this->tileDataNum);
            else{
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = this->tileDataNum * sizeof(TYPE_X);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPad(yGm[progress * this->tileDataNum], yLocal,copyParams );
            }
            inQueueX.FreeTensor(xLocal);
            outQueueY.FreeTensor(yLocal);
        }

        __aicore__ inline void CopyIn2(uint32_t b,uint32_t b1,uint32_t w){
            LocalTensor<TYPE_X> xLocal = inQueuedX.AllocTensor<TYPE_X>();
            event_t eventIDMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            auto in = b*this->tileDataNum+w*this->dim3*this->dim4*this->dim5+b1*this->dim4*this->dim5;
            if((this->dim4*this->dim5*sizeof(TYPE_X))%32==0){
                DataCopy(xLocal,xGm[in],this->dim4*this->dim5);
                SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            }
            else{
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = this->dim5*this->dim4 * sizeof(TYPE_X);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPadExtParams<TYPE_X> padParams{true, 0, 0, 0};
                DataCopyPad(xLocal, xGm[in],copyParams ,padParams);
                SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            }
            auto out = b*this->tileDataNum+b1*this->dim2*this->dim5*this->dim4+w*this->dim4*this->dim5;
            if((this->dim5*this->dim4*sizeof(TYPE_X))%32!=0){
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = this->dim5*this->dim4* sizeof(TYPE_X);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
                DataCopyPad(yGm[out],xLocal,copyParams );
            }else{
                WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);         
                DataCopy(yGm[out], xLocal, this->dim5*this->dim4);
            }
            inQueuedX.FreeTensor(xLocal); 

        }

//         __aicore__ inline void CopyOut2(uint32_t b,uint32_t b1,uint32_t w){
//             LocalTensor<TYPE_X> xLocal = inQueuedX.DeQue<TYPE_X>();
//             DataCopy(yGm[progress*this->tileDataNum+w*this->dim3*this->dim4*this->dim5+bs*this->dim4*this->dim5],xLocal,this->dim4*this->dim5);
//             inQueuedX.FreeTensor(xLocal);
//         }
    private:
        TPipe pipe;
        TQue<QuePosition::VECIN, 2> inQueueX;
        TQueBind<TPosition::VECIN,TPosition::VECOUT,2> inQueuedX;
        TQue<QuePosition::VECOUT, 2> outQueueY;
        GlobalTensor<TYPE_X> xGm;
        GlobalTensor<TYPE_Y> yGm;
        uint32_t CoreDataNum,CoreBlockNum,tileDataNum;
        uint32_t dim1,dim2,dim3,dim4,dim5,Batch;
        uint32_t DoubleBuffer,Type,Pad;
};

extern "C" __global__ __aicore__ void depth_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelDepthToSpace<DTYPE_X,DTYPE_Y> op;
    // printf("Kernel Start\n");
    op.Init(x,y,tiling_data.smallCoreBlockNum,tiling_data.bigCoreBlockNum,tiling_data.tileDataNum,
            tiling_data.tailBlockNum,tiling_data.dim1,tiling_data.dim2,tiling_data.dim3,tiling_data.dim4,
            tiling_data.dim5,tiling_data.Pad,tiling_data.Batch,tiling_data.DoubleBuffer,tiling_data.Type);
    if (tiling_data.Type==0)
    {
        op.Process0();
    }else if(tiling_data.Type==1){
        op.Process1();
    }
    else if(tiling_data.Type==3){
        op.Process3();
    }else{
        op.Process2();
    }
    
}