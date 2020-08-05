
set lang=wiki_sql

set beam_size=10 ^
set batch_size=128 ^
set source_length=256 ^
set target_length=128 ^
set output_dir=model/%lang% ^
set data_dir=../data/test/ ^
set dev_file=%data_dir%/%lang%/valid.jsonl ^
test_file=%data_dir%/%lang%/test.jsonl ^
test_model=%output_dir%/checkpoint-best-wiki/pytorch_model.bin ^
:: checkpoint for test ^

python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path %test_model% --dev_filename %dev_file% --test_filename %test_file% --output_dir %output_dir% --max_source_length %source_length% --max_target_length %target_length% --beam_size %beam_size% --eval_batch_size %batch_size%