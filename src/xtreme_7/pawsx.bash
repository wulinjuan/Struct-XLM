for a in 2 1
do
  for lr in 8e-6 9e-6 1e-5 2e-5
  do
    for epoch in 5 10
    do
      for warmup in 0.0 0.05 0.1
      do
        for wd in 0.0
        do python run_classify_pawsx.py --save_only_best_checkpoint --do_predict --do_train \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'output/pawsx' \
--critic_output_dir '/data/home10b/wlj/code/Struct-XLM/src/Struct_XLM/output_struct_xlm/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps $a \
--learning_rate $lr  \
--weight_decay $wd \
--num_train_epochs $epoch \
--warmup_steps  $warmup
done
done
done
done
done
