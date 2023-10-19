for a in 2
do
  for lr in 7e-6 2e-5
  do
    for epoch in 10
    do
      for warmup in 0.0 0.05 0.1
      do
        for wd in 0.0
        do python run_tag_ner.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'data/NER' \
--labels 'data/NER/label.txt' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'output/NER' \
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

for a in 1
do
  for lr in 8e-6 9e-6 1e-5 7e-6 2e-5
  do
    for epoch in 10
    do
      for warmup in 0.0 0.05 0.1
      do
        for wd in 0.0
        do python run_tag_ner.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'data/NER' \
--labels 'data/NER/label.txt' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'output/NER' \
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
