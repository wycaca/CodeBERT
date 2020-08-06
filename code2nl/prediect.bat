cd code2nl
set lang=wiki_sql

set beam_size=10
set batch_size=64
set source_length=256
set target_length=128
set output_dir=model/%lang%
set data_dir=./CodeSearchNet/%lang%
set test_file=%data_dir%/test.jsonl
set test_model=%output_dir%/checkpoint-best-bleu/pytorch_model.bin

python prediect.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path %test_model%  ^
--data_dir %data_dir% ^
--test_filename %test_file% --output_dir %output_dir% ^
--max_source_length %source_length% --max_target_length %target_length% ^
--beam_size %beam_size% --eval_batch_size %batch_size%