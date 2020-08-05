cd code2nl

set lang=wiki_sql
:: set lr=5e-5
set lr=5e-5
set batch_size=6
set beam_size=10
set source_length=256
set target_length=128
set data_dir=../code2nl/CodeSearchNet
set output_dir=model/%lang%
set train_file=%data_dir%/%lang%/train.jsonl
set dev_file=%data_dir%/%lang%/valid.jsonl
::400 for ruby, 600 for javascript, 1000 for others
set eval_steps=600
::20000 for ruby, 30000 for javascript, 50000 for others
set train_steps=30000
::Roberta: roberta-base
set pretrained_model=microsoft/codebert-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path %pretrained_model% ^
--train_filename %train_file% --dev_filename %dev_file% ^
--output_dir %output_dir% --max_source_length %source_length% --max_target_length %target_length% ^
--beam_size %beam_size% --train_batch_size %batch_size% --eval_batch_size %batch_size% ^
--learning_rate %lr% --train_steps %train_steps% --eval_steps %eval_steps%