##   ex) bash train.sh


export CUDA_VISIBLE_DEVICES=4




cmd="${cmd}python test_reader.py   
            --model_path 
            --model_name google/flan-t5-base
            --write_results   --eval_data 
            --per_gpu_batch_size 1        --n_context 50       --name my_test    --text_maxlength 300    
            --checkpoint_dir checkpoint/test/seed3613_ncontext50_len300_batch4_acc2_trecdl19"


echo $cmd
$cmd
