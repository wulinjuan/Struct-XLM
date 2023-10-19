# train the Struct-XLM
python src/Struct_XLM/run_main.py --do_train --do_eval \
--output_dir 'output_struct-xlm/' \
--data_dir 'data/UD_data/' \
--sentence_type 'mean' \
--ml_type 'xlmr' \
--ml_model_path '/home/shenyl/cached_models/xlm-roberta-large' \
--max_len 256 \
--margin 0.4 \
--act_dropout 0.5 \
--tau 0.5 \
--alpha 0.5 \
--learning_rate 6e-6 \

# script for seven XTREME Benckmark tasks
# 1、Training XNLI and evaluation
python src/xtreme_7/run_classify_XNLI.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/XNLI' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/XNLI' \
--critic_output_dir 'output_struct-xlm/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 8e-6  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  5000 \

# 2、Training PAWS-X and evaluation
python src/xtreme_7/run_classify_pawsx.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/PAWS-X' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/paws-x' \
--critic_output_dir 'output_struct-xlm/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 1e-5  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1 \


# 3、Training NER and evaluation
python src/xtreme_7/run_tag_ner.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/NER' \
--labels 'src/xtreme_7/data/NER/label.txt' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/NER' \
--critic_output_dir 'output_struct-xlm/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 9e-6  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1 \

# 4、Training POS and evaluation
python src/xtreme_7/run_tag_udpos.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/udpos' \
--labels 'src/xtreme_7/data/udpos/label.txt' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/udpos' \
--critic_output_dir 'output_struct-xlm/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 9e-6  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1 \

# 5、Training XQuAD and evaluation
python src/xtreme_7/run_squad.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/MRC' \
--test_file 'xquad' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/xquad' \
--critic_output_dir 'output_struct-xlm/' \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-5  \
--weight_decay 0.0 \
--num_train_epochs 3.0 \
--warmup_steps  0.1 \

# 6、Training MLQA and evaluation
python src/xtreme_7/run_squad.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/MRC' \
--test_file 'mlqa' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/mlqa' \
--critic_output_dir 'output_struct-xlm/' \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-5  \
--weight_decay 0.0 \
--num_train_epochs 2.0 \
--warmup_steps  0.1 \

# 7、Training TyDiQA and evaluation
python src/xtreme_7/run_tydiqa.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/MRC' \
--test_file 'tydiqa' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/tydiqa' \
--critic_output_dir 'output_struct-xlm/' \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-5  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1 \

