# TTNet-Pytorch
The implementation for the paper "TTNet: Real-time temporal and spatial video analysis of table tennis"

---

## 1. Features
- [x] Ball detection global stage
- [x] Ball detection local stage (refinement)
- [x] Events Spotting detection (Bounce and Net hit)
- [x] Semantic Segmentation (Human, table, and scoreboard)
- [x] [Multi-Task learning](https://arxiv.org/pdf/1705.07115.pdf)
- [x] [Distributed Data Parallel Training](https://github.com/pytorch/examples/tree/master/distributed/ddp)
- [ ] Evaluate

## 2. Getting Started

### 2.1. Preparing the dataset
The instruction for the dataset preparation is [here](./prepare_dataset/README.md)

### 2.2. How to run

#### 2.2.1. Training
```shell script
cd src/training/
```
##### 2.2.1.1. Single machine, single gpu

```shell script
python main.py --gpu_idx 0
```

##### 2.2.1.2. Multi-processing Distributed Data Parallel Training
We should always use the NCCL backend for multi-processing distributed training since it currently provides the best 
distributed training performance.

- **Single machine (node), multiple GPUs**

```shell script
python main.py --dist-url 'tcp://127.0.0.1:29500' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```

- **Multiple machines (nodes), multiple GPUs**

_**First machine**_

```shell script
python main.py --dist-url 'tcp://127.0.0.1:29500' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0
```
_**Second machine**_

```shell script
python main.py --dist-url 'tcp://127.0.0.1:29500' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1
```

#### 2.2.2. Evaluation
    
## Requirements
Refer [the tutorial](https://github.com/maudzung/virtual_environment_python3) to install a virtual environment

The source code will be updated soon... Staring and watching us...