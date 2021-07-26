gpus=7

'''
CUDA_VISIBLE_DEVICES=$gpus python3 run_lm.py
	--output_dir ./models/EsperBERTo-small-v1
	--model_type roberta
	--mlm
	--tokenizer_name roberta-based
	--do_train
	--learning_rate 1e-4
	--num_train_epochs 5
	--save_total_limit 2
	--save_steps 2000
	--per_gpu_train_batch_size 4
	--evaluate_during_training
	--seed 42
	--train_data_file eo-dedup-train.txt
'''


TRAIN_FILE=data/wikitext-103-raw/wiki.train.raw
TEST_FILE=data/wikitext-103-raw/wiki.test.raw
OUTPUT_DIR=result_mlm
MODEL_TYPE=roberta
MODEL_NAME_OR_PATH=roberta-base


CUDA_VISIBLE_DEVICES=$gpus python3 run_lm.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=$MODEL_TYPE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm
