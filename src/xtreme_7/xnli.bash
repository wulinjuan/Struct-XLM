for a in 4 2
do
  for lr in 9e-6 8e-6
  do
    for epoch in 10
    do
      for warmup in 12500 500
      do
        for wd in 0.0
        do python run_classify_XNLI.py --save_only_best_checkpoint --do_predict --do_train \
--model_type 'xlmr' \
--model_name_or_path '/data/home10b/wlj/cached_models/xlm-roberta-large' \
--output_dir 'output/xnli' \
--critic_output_dir '/data/home10b/wlj/code/Struct-XLM/src/Struct_XLM/output_no_warm_start/' \
--per_gpu_train_batch_size 8 \
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
