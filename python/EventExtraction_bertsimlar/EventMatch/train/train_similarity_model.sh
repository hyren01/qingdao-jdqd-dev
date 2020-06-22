#!/usr/bin bash

TRAIN_DATA_PATH=./resources/dataset/data/train.txt
VALID_DATA_PATH=./resources/dataset/data/dev.txt
TEST_DATA_PATH=./resources/dataset/data/test.txt
CONFIG_PATH=./model/initial_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json
INIT_CHECKPONIT=./model/initial_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt
VOCAB_PATH=./model/initial_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt
TRAINED_MODEL_PATH=./model/trained_model/best_val_acc_model.h5

python3 train_similarity_model.py \
		--batch_size 10 \
		--epoch 3 \
		--max_length 256 \
		--gelu tanh \
		--train_data_dir ${TRAIN_DATA_PATH} \
		--valid_data_dir ${VALID_DATA_PATH} \
		--test_data_dir ${TEST_DATA_PATH} \
		--config_path ${CONFIG_PATH} \
		--init_checkpoint ${INIT_CHECKPONIT} \
		--vocab_path ${VOCAB_PATH}\
		--trained_model_path ${TRAINED_MODEL_PATH}
