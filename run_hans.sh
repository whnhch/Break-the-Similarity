export WANDB_ENTITY="team-hada"
export WANDB_PROJECT="hans_test_wh"
files=`ls -d ./result/supcon_bt_break_the_sim*`

for file in $files
do
    echo "testing "$file"..."
    python evaluation.py \
        --model_name_or_path $file \
        --pooler cls \
        --task_set hans \
        --mode test
done
