#include "kernel_operator.h"
#include <vector>
const int DoubleBuffer = 2;
using namespace AscendC;
template<typename DTYPE> class KernelGroupNormV2{
    
    public:
        __aicore__ inline KernelGroupNormV2() {}
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,float epsilon,
                                    uint32_t groupSize,uint32_t channelSize,uint32_t Batch,uint32_t num_groups,
                                    uint32_t Channel,uint32_t totalSize,uint32_t tileDataNum,uint32_t tailDataNum,uint32_t tileNum,
                                    uint32_t tileChannelData,uint32_t tailChannelData,uint32_t tileChannelNum) {
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            uint32_t coreNum = GetBlockIdx();
            uint32_t globalBufferIndex,globalChannelIndex;
            
            this->num_groups = num_groups;
            this->Channel = Channel;
            this->group_c = Channel / num_groups;
            this->Batch = Batch;
            this->groupSize = groupSize;
            this->channelSize = channelSize;
            this->epsilon = epsilon;
            this->tileDataNum = tileDataNum;
            this->tailDataNum = tailDataNum;
            this->tileNum = tileNum;
            this->tileChannelData = tileChannelData;
            this->tailChannelData = tailChannelData;
            this->tileChannelNum = tileChannelNum;

            xGm.SetGlobalBuffer((__gm__ DTYPE*)x, totalSize);
            yGm.SetGlobalBuffer((__gm__ DTYPE*)y, totalSize);
            gammaGm.SetGlobalBuffer((__gm__ DTYPE*)gamma,Channel);
            betaGm.SetGlobalBuffer((__gm__ DTYPE*)beta,Channel);
            meanGm.SetGlobalBuffer((__gm__ DTYPE*)mean,Batch*num_groups);
            rstdGm.SetGlobalBuffer((__gm__ DTYPE*)rstd,Batch*num_groups);
            // if(channelSize * sizeof(DTYPE)%256==0){
                // printf("pipe init");
                // pipe.InitBuffer(inQueueGX, DoubleBuffer, groupSize * sizeof(DTYPE));
                pipe.InitBuffer(inQueueCX, DoubleBuffer, tileDataNum * sizeof(DTYPE));
                pipe.InitBuffer(inQueuedX, DoubleBuffer, tileChannelData * sizeof(DTYPE));
            // }
        }
        __aicore__ inline void Process(){
            // printf("Batch = %d,num_groups = %d,groupSize = %d",Batch,num_groups,groupSize);
            for(uint32_t i=0;i<Batch;i++)
                for(uint32_t j=0;j<num_groups;j++){
                    float sum = 0.0;
                    for(uint32_t k=0;k<groupSize;k++){
                        float val = xGm.GetValue(i*num_groups*groupSize+j*groupSize+k);
                        sum+=val;
                    }
                    float avg = sum / groupSize;
                    meanGm.SetValue(i*num_groups+j,(DTYPE)avg);
                }
            
            for(uint32_t i=0;i<Batch;i++)
                for(uint32_t j=0;j<num_groups;j++){
                    float avg = meanGm.GetValue(i*num_groups+j);
                    float sum = 0.0;
                    for(uint32_t k=0;k<groupSize;k++){
                        float val = xGm.GetValue(i*num_groups*groupSize+j*groupSize+k);
                        sum+=(val-avg) * (val-avg);
                    }
                    float var = sum / groupSize;
                    // float rstd = 1.0f / sqrt(var + epsilon);
                    rstdGm.SetValue(i*num_groups+j,(DTYPE)var);
                }

            for(uint32_t i=0;i<Batch;i++)
                for(uint32_t j=0;j<num_groups;j++){
                    float mean = meanGm.GetValue(i*num_groups+j);
                    float var = rstdGm.GetValue(i*num_groups+j);
                    for(uint32_t k=0;k<group_c;k++){
                        float gamma = gammaGm.GetValue(j*group_c+k);
                        float beta = betaGm.GetValue(j*group_c+k);
                        float sum = 0.0;
                        for(uint32_t l = 0;l<channelSize;l++){
                            auto index = i*num_groups*group_c*channelSize+j*group_c*channelSize+k*channelSize+l;
                            float x = xGm.GetValue(index);
                            float result = gamma * ((x-mean) / sqrt(var + epsilon)) + beta;
                            // printf("%f ",result);
                            yGm.SetValue(index,(DTYPE)result);
                        }
                    }
                }
        }
    
        __aicore__ inline void Process1(){
            // printf("tileDataNum = %d \n",tileDataNum);
            // printf("tailDataNum = %d \n",tailDataNum);
            // printf("tileNum = %d \n",tileNum);
            // printf("tileChannelNum = %d \n",tileChannelNum);
            // printf("tileChannelData = %d \n",tileChannelData);
            // auto cof = T(1.0f / groupSize);
            // LocalTensor<DTYPE> y = outQueueY.AllocTensor<DTYPE>();
            for(uint32_t i=0;i<Batch;i++)
                for(uint32_t j=0;j<num_groups;j++){
                    this->processDataNum = this->tileDataNum;
                    float result = 0.0;
                    for(uint32_t k=0;k<this->tileNum;k++){
                        if(k==this->tileNum-1) this->processDataNum = this->tailDataNum;
                        {
                            LocalTensor<DTYPE> x = inQueueCX.AllocTensor<DTYPE>();
                            DataCopy(x, xGm[i*num_groups*this->groupSize+j*this->groupSize+k*this->tileDataNum], this->processDataNum);
                            inQueueCX.EnQue(x);
                        // inQueueCX.FreeTensor(x);
                        }
                        {
                            uint32_t  dataSize = this->processDataNum;
                            LocalTensor<DTYPE> x = inQueueCX.DeQue<DTYPE>();
                            // LocalTensor<DTYPE> y = outQueueY.AllocTensor<DTYPE>();
                            uint32_t reduceCount = 256/sizeof(DTYPE);
                            if(dataSize / reduceCount > 1){
                                uint32_t repeatTimes = dataSize / reduceCount;
                                uint32_t repStride = 256/32;
                                WholeReduceSum(x, x, reduceCount, repeatTimes, 1, 1, repStride);
                                dataSize /=reduceCount;
                            }
                            // outQueueY.EnQue(y);
                            WholeReduceSum(x,x,dataSize,1,1,1,0);
                            event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                            SetFlag<HardEvent::V_S>(eventIDSToMTE3);
                            WaitFlag<HardEvent::V_S>(eventIDSToMTE3);
                            float mid = x.GetValue(0);
                            // printf("%f ",mid);
                            result += mid;
                            inQueueCX.FreeTensor(x);
                        }
                    }
                    result /= groupSize;
                    meanGm.SetValue(i*num_groups+j,(DTYPE)result);
                }
            for(uint32_t i=0;i<Batch;i++)
                for(uint32_t j=0;j<num_groups;j++){
                    this->processDataNum = this->tileDataNum;
                    float result = 0.0;
                    // printf("\nmean = %f\n",meanGm.GetValue(i*num_groups+j));
                    float mean = meanGm.GetValue(i*num_groups+j);
                    for(uint32_t k=0;k<this->tileNum;k++){
                        if(k==this->tileNum-1) this->processDataNum = this->tailDataNum;
                        {
                            LocalTensor<DTYPE> x = inQueueCX.AllocTensor<DTYPE>();
                            DataCopy(x, xGm[i*num_groups*this->groupSize+j*this->groupSize+k*this->tileDataNum], this->processDataNum);
                            inQueueCX.EnQue(x);
                        }
                        {
                            uint32_t  dataSize = this->processDataNum;
                            LocalTensor<DTYPE> x = inQueueCX.DeQue<DTYPE>();
                            Adds(x,x,DTYPE(-mean),this->processDataNum);
                            Mul(x, x, x, this->processDataNum);
                            uint32_t reduceCount = 256/sizeof(DTYPE);
                            if(dataSize / reduceCount > 1){
                                uint32_t repeatTimes = dataSize / reduceCount;
                                uint32_t repStride = 256/32;
                                WholeReduceSum(x, x, reduceCount, repeatTimes, 1, 1, repStride);
                                dataSize /=reduceCount;
                            }
                            WholeReduceSum(x,x,dataSize,1,1,1,0);
                            event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                            SetFlag<HardEvent::V_S>(eventIDSToMTE3);
                            WaitFlag<HardEvent::V_S>(eventIDSToMTE3);
                            float mid = x.GetValue(0);
                            // printf("%f ",mid);
                            result += mid;
                            inQueueCX.FreeTensor(x);
                        }
                    }
                    result /= groupSize;
                    // printf("\n%f\n",result);
                    rstdGm.SetValue(i*num_groups+j,(DTYPE)result);
                }
            for(uint32_t i=0;i<Batch;i++)
                for(uint32_t j=0;j<num_groups;j++){
                    float mean = meanGm.GetValue(i*num_groups+j);
                    float var = rstdGm.GetValue(i*num_groups+j);
                    float deno = 1.0f / sqrt(var + this->epsilon);
                    for(uint32_t k=0;k<group_c;k++){
                        this->processDataNum = this->tileChannelData;
                        float gamma = gammaGm.GetValue(j*group_c+k);
                        float beta = betaGm.GetValue(j*group_c+k);
                        // for(uint32_t l = 0;l<channelSize;l++){
                        //     auto index = i*num_groups*group_c*channelSize+j*group_c*channelSize+k*channelSize+l;
                        //     float x = xGm.GetValue(index);
                        //     float result = gamma * ((x-mean) * deno) + beta;
                        //     // printf("%f ",result);
                        //     yGm.SetValue(index,(DTYPE)result);
                        // }
                        for(uint32_t l=0;l<tileChannelNum;l++){
                            if(l==this->tileChannelNum-1) this->processDataNum = this->tailChannelData;
                            
                                LocalTensor<DTYPE> x = inQueuedX.AllocTensor<DTYPE>();
                                DataCopy(x, xGm[i*num_groups*this->groupSize+j*this->groupSize+k*this->channelSize+l*this->tileChannelData], this->processDataNum);
                                // for (int m=0;m<x.GetSize();m++)
                                //     printf("%f ",x.GetValue(m));
                                // inQueuedX.EnQue(x);
                            
                            
                                // LocalTensor<DTYPE> x = inQueuedX.DeQue<DTYPE>();
                                // printf("%d ",x.GetSize());
                            //     for (int m=0;m<x.GetSize();m++)
                            //         printf("%f ",x.GetValue(m));
                                // printf("error\n");
                                event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                                Adds(x,x,DTYPE(-mean),this->processDataNum);
                                Muls(x,x,(DTYPE)deno,this->processDataNum);
                                Muls(x,x,(DTYPE)gamma,this->processDataNum);
                                Adds(x,x,(DTYPE)beta,this->processDataNum);
                                event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                                SetFlag<HardEvent::V_MTE3>(eventIDSToMTE3);
                                WaitFlag<HardEvent::V_MTE3>(eventIDSToMTE3);
                                DataCopy(yGm[i*num_groups*this->groupSize+j*this->groupSize+k*this->channelSize+l*this->tileChannelData],x,this->processDataNum);
                                inQueuedX.FreeTensor(x);
                            
                        }
                    }
                
                }
            
        }




    private:
        TPipe pipe;
        TQue<QuePosition::VECIN, 2> inQueueCX;
        TQueBind<TPosition::VECIN,TPosition::VECOUT,2> inQueuedX;
        // TQue<QuePosition::VECOUT, 2> outQueueY;
        GlobalTensor<DTYPE> xGm;
        GlobalTensor<DTYPE> yGm;
        GlobalTensor<DTYPE> gammaGm;
        GlobalTensor<DTYPE> betaGm;
        GlobalTensor<DTYPE> meanGm;
        GlobalTensor<DTYPE> rstdGm;
        uint32_t groupSize,channelSize;
        uint32_t num_groups,Channel,Batch,group_c;
        uint32_t tailDataNum,tileDataNum,tileNum,processDataNum;
        uint32_t tileChannelData,tailChannelData,tileChannelNum;
        float epsilon;
};


template<typename DTYPE> class KernelGroupNormV2_fast{
    
    public:
        __aicore__ inline KernelGroupNormV2_fast() {}
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,float epsilon,
                                    uint32_t groupSize,uint32_t channelSize,uint32_t Batch,uint32_t num_groups,
                                    uint32_t Channel,uint32_t totalSize,uint32_t tileDataNum,uint32_t tailDataNum,uint32_t tileNum,
                                    uint32_t tileChannelData,uint32_t tailChannelData,uint32_t tileChannelNum,
                                    uint32_t bigCoreBlockNum,uint32_t smallCoreBlockNum,uint32_t tailBlockNum) {
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            uint32_t coreNum = GetBlockIdx();
            uint32_t globalBufferIndex,groupIndex;
            
            this->num_groups = num_groups;
            this->Channel = Channel;
            this->group_c = Channel / num_groups;
            this->Batch = Batch;
            this->groupSize = groupSize;
            this->channelSize = channelSize;
            this->epsilon = epsilon;
            this->tileDataNum = tileDataNum;
            this->tailDataNum = tailDataNum;
            this->tileNum = tileNum;
            this->tileChannelData = tileChannelData;
            this->tailChannelData = tailChannelData;
            this->tileChannelNum = tileChannelNum;
            // this->tileDataNum = this->group_c*H*W;
            if (coreNum < tailBlockNum) {
                this->coreBlockNum = bigCoreBlockNum;
            //     this->CoreDataNum = bigCoreBlockNum*tileDataNum;
            //     this->channelNum = bigCoreBlockNum * group_c;
                globalBufferIndex = bigCoreBlockNum*this->groupSize * GetBlockIdx();
                this->groupIndex = bigCoreBlockNum * GetBlockIdx();
            //     globalChannelIndex = this->channelNum*GetBlockIdx();
            }else{
                this->coreBlockNum = smallCoreBlockNum;
            //     this->CoreDataNum = smallCoreBlockNum*tileDataNum;
            //     this->chanelNum = smallCoreBlockNum * group_c;
                globalBufferIndex = tailBlockNum *bigCoreBlockNum*this->groupSize
                                    + (GetBlockIdx() - tailBlockNum)*this->groupSize*smallCoreBlockNum;
                this->groupIndex = tailBlockNum *bigCoreBlockNum+ (GetBlockIdx() - tailBlockNum)*smallCoreBlockNum;
            //     globalChannelIndex = tailBlockNum *bigCoreBlockNum*group_c 
            //                         + (GetBlockIdx() - tailBlockNum)*group_c*smallCoreBlockNum;
            }
            xGm.SetGlobalBuffer((__gm__ DTYPE*)x+globalBufferIndex, coreBlockNum*groupSize);
            yGm.SetGlobalBuffer((__gm__ DTYPE*)y+globalBufferIndex, coreBlockNum*groupSize);
            gammaGm.SetGlobalBuffer((__gm__ DTYPE*)gamma,Channel);
            betaGm.SetGlobalBuffer((__gm__ DTYPE*)beta,Channel);
            meanGm.SetGlobalBuffer((__gm__ DTYPE*)mean+this->groupIndex,coreBlockNum);
            rstdGm.SetGlobalBuffer((__gm__ DTYPE*)rstd+this->groupIndex,coreBlockNum);
            pipe.InitBuffer(inQueueCX, DoubleBuffer, tileDataNum * sizeof(DTYPE));
            pipe.InitBuffer(inQueuedX, DoubleBuffer, tileChannelData * sizeof(DTYPE));
        }
    
        __aicore__ inline void Process(){
                for(uint32_t j=0;j<coreBlockNum;j++){
                    this->processDataNum = this->tileDataNum;
                    float result = 0.0;
                    for(uint32_t k=0;k<this->tileNum;k++){
                        if(k==this->tileNum-1) this->processDataNum = this->tailDataNum;
                        {
                            LocalTensor<DTYPE> x = inQueueCX.AllocTensor<DTYPE>();
                            DataCopy(x, xGm[j*this->groupSize+k*this->tileDataNum], this->processDataNum);
                            inQueueCX.EnQue(x);
                        }
                        {
                            uint32_t  dataSize = this->processDataNum;
                            LocalTensor<DTYPE> x = inQueueCX.DeQue<DTYPE>();
                            uint32_t reduceCount = 256/sizeof(DTYPE);
                            if(dataSize / reduceCount > 1){
                                uint32_t repeatTimes = dataSize / reduceCount;
                                uint32_t repStride = 256/32;
                                WholeReduceSum(x, x, reduceCount, repeatTimes, 1, 1, repStride);
                                dataSize /=reduceCount;
                            }
                            WholeReduceSum(x,x,dataSize,1,1,1,0);
                            event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                            SetFlag<HardEvent::V_S>(eventIDSToMTE3);
                            WaitFlag<HardEvent::V_S>(eventIDSToMTE3);
                            float mid = x.GetValue(0);
                            result += mid;
                            inQueueCX.FreeTensor(x);
                        }
                    }
                    result /= groupSize;
                    meanGm.SetValue(j,(DTYPE)result);
                }
                for(uint32_t j=0;j<coreBlockNum;j++){
                    this->processDataNum = this->tileDataNum;
                    float result = 0.0;
                    float mean = meanGm.GetValue(j);
                    for(uint32_t k=0;k<this->tileNum;k++){
                        if(k==this->tileNum-1) this->processDataNum = this->tailDataNum;
                        {
                            LocalTensor<DTYPE> x = inQueueCX.AllocTensor<DTYPE>();
                            DataCopy(x, xGm[j*this->groupSize+k*this->tileDataNum], this->processDataNum);
                            inQueueCX.EnQue(x);
                        }
                        {
                            uint32_t  dataSize = this->processDataNum;
                            LocalTensor<DTYPE> x = inQueueCX.DeQue<DTYPE>();
                            Adds(x,x,DTYPE(-mean),this->processDataNum);
                            Mul(x, x, x, this->processDataNum);
                            uint32_t reduceCount = 256/sizeof(DTYPE);
                            if(dataSize / reduceCount > 1){
                                uint32_t repeatTimes = dataSize / reduceCount;
                                uint32_t repStride = 256/32;
                                WholeReduceSum(x, x, reduceCount, repeatTimes, 1, 1, repStride);
                                dataSize /=reduceCount;
                            }
                            WholeReduceSum(x,x,dataSize,1,1,1,0);
                            event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                            SetFlag<HardEvent::V_S>(eventIDSToMTE3);
                            WaitFlag<HardEvent::V_S>(eventIDSToMTE3);
                            float mid = x.GetValue(0);
                            result += mid;
                            inQueueCX.FreeTensor(x);
                        }
                    }
                    result /= groupSize;
                    rstdGm.SetValue(j,(DTYPE)result);
                }
                for(uint32_t j=0;j<coreBlockNum;j++){
                    float mean = meanGm.GetValue(j);
                    float var = rstdGm.GetValue(j);
                    float deno = 1.0f / sqrt(var + this->epsilon);
                    for(uint32_t k=0;k<group_c;k++){
                        this->processDataNum = this->tileChannelData;
                        float gamma = gammaGm.GetValue(((groupIndex+j)%num_groups)*group_c+k);
                        float beta = betaGm.GetValue(((groupIndex+j)%num_groups)*group_c+k);
                        for(uint32_t l=0;l<tileChannelNum;l++){
                            if(l==this->tileChannelNum-1) this->processDataNum = this->tailChannelData;
                            
                                LocalTensor<DTYPE> x = inQueuedX.AllocTensor<DTYPE>();
                                DataCopy(x, xGm[j*this->groupSize+k*this->channelSize+l*this->tileChannelData], this->processDataNum);
                                event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                                Adds(x,x,DTYPE(-mean),this->processDataNum);
                                Muls(x,x,(DTYPE)deno,this->processDataNum);
                                Muls(x,x,(DTYPE)gamma,this->processDataNum);
                                Adds(x,x,(DTYPE)beta,this->processDataNum);
                                event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                                SetFlag<HardEvent::V_MTE3>(eventIDSToMTE3);
                                WaitFlag<HardEvent::V_MTE3>(eventIDSToMTE3);
                                DataCopy(yGm[j*this->groupSize+k*this->channelSize+l*this->tileChannelData],x,this->processDataNum);
                                inQueuedX.FreeTensor(x);
                            
                        }
                    }
                
                }
            
        }




    private:
        TPipe pipe;
        TQue<QuePosition::VECIN, 2> inQueueCX;
        TQueBind<TPosition::VECIN,TPosition::VECOUT,2> inQueuedX;
        // TQue<QuePosition::VECOUT, 2> outQueueY;
        GlobalTensor<DTYPE> xGm;
        GlobalTensor<DTYPE> yGm;
        GlobalTensor<DTYPE> gammaGm;
        GlobalTensor<DTYPE> betaGm;
        GlobalTensor<DTYPE> meanGm;
        GlobalTensor<DTYPE> rstdGm;
        uint32_t groupSize,channelSize;
        uint32_t num_groups,Channel,Batch,group_c;
        uint32_t tailDataNum,tileDataNum,tileNum,processDataNum;
        uint32_t tileChannelData,tailChannelData,tileChannelNum;
        uint32_t coreBlockNum,groupIndex;
        float epsilon;
};



extern "C" __global__ __aicore__ void group_norm_v2(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    // if(tiling_data.channelSize*sizeof(DTYPE_X)%256!=0){
    if(tiling_data.groupSize*sizeof(DTYPE_X)%256!=0){
        KernelGroupNormV2<DTYPE_X> op;
        op.Init(x,gamma,beta,y,mean,rstd,tiling_data.epsilon,tiling_data.groupSize,tiling_data.channelSize,tiling_data.Batch,tiling_data.num_groups,tiling_data.Channel,tiling_data.totalSize,tiling_data.tileDataNum,tiling_data.tailDataNum,tiling_data.tileNum,tiling_data.tileChannelData,tiling_data.tailChannelData,tiling_data.tileChannelNum);
    
        op.Process();
    }
    else {
        KernelGroupNormV2_fast<DTYPE_X> op;
        op.Init(x,gamma,beta,y,mean,rstd,tiling_data.epsilon,tiling_data.groupSize,tiling_data.channelSize,tiling_data.Batch,tiling_data.num_groups,tiling_data.Channel,tiling_data.totalSize,tiling_data.tileDataNum,tiling_data.tailDataNum,tiling_data.tileNum,tiling_data.tileChannelData,tiling_data.tailChannelData,tiling_data.tileChannelNum,tiling_data.bigCoreBlockNum,tiling_data.smallCoreBlockNum,tiling_data.tailBlockNum);
        op.Process();
    }
}