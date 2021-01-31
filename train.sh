export PYTHONPATH=./:/home/zhfeing/project/deep-learning-lib:$PYTHONPATH
export TORCH_HOME=/home/zhfeing/model-zoo


export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_distributed.py \
    --num-nodes 1 \
    --rank 0 \
    --multiprocessing \
    --dist-url tcp://localhost:9001 \
    --log-dir run/distributed-with-syncbn \
    --file-name-cfg ResNet50 \
    --cfg-filepath config/ResNet50.yml \
    --seed 1029 &
