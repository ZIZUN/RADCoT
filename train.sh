##   ex) bash train.sh 4
ngpu_ddp=$1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=${ngpu_ddp}

# (DDP)

cmd="${cmd}python -m torch.distributed.launch --nproc_per_node=${ngpu_ddp} --master_port=35222"

cmd="${cmd} train_reader.py 
            --train_data train.json
            --eval_data dev.json
            --model_size small          --per_gpu_batch_size 1          --n_context 10
            --name my_experiment   --text_maxlength 300   --seed 3613 --accumulation_steps 2
            --eval_freq 500
            --save_freq 1000
            --total_steps 100000
            --model_path none 
            --checkpoint_dir checkpoint/train/seed3613_ncontext10_len300_batch4_acc2_small
            "

echo $cmd
$cmd
