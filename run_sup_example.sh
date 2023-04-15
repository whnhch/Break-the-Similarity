NUM_GPU=2

PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"

file_name=data/supcon_de_aeda.csv

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path roberta-base \
    --train_file ${file_name} \
    --output_dir result/${file_name} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --lmbd1 0.6 \
    --lmbd2 0.9 \
    --fp16 \
    "$@"