@echo off

cd codesearch

:: fine-tuning a language-specific model for each programming language
set lang=sql
:: CodeBERT: path to .bin file. Roberta: roberta-base
set pretrained_model=../pretrained_models/pytorch_model.bin

python run_classifier.py ^
--model_type roberta ^
--task_name codesearch ^
--do_train ^
--do_eval ^
--eval_all_checkpoints ^
--train_file train.txt ^
--dev_file valid.txt ^
--max_seq_length 200 ^
--per_gpu_train_batch_size 20 ^
--per_gpu_eval_batch_size 20 ^
--learning_rate 5e-6 ^
--num_train_epochs 3 ^
--gradient_accumulation_steps 1 ^
--overwrite_output_dir ^
--data_dir ../data/train_valid/%lang% ^
--output_dir ./models/%lang%  ^
--model_name_or_path %pretrained_model% ^
--config_name roberta-base