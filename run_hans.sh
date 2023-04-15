export WANDB_ENTITY="team-hada"
export WANDB_PROJECT="hans_test_wh"
files=`ls -d ./result/pecE_bt_original_case8_0.8*`

for file in $files
do
    echo "testing "$file"..."
    python evaluation.py \
        --model_name_or_path $file \
        --pooler cls \
        --task_set hans \
        --mode test
done