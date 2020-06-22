#!/usr/bin bash

TRAIN_DATA_PATH=./resources/dataset/data_second/train.txt
VALID_DATA_PATH=./resources/dataset/data_second/dev.txt
TEST_DATA_PATH=./resources/dataset/data_second/test.txt
VOCAB_PATH=./initial_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt
TRAINED_MODEL_PATH=./model/trained_model/best_val_acc_model.h5
SECOND_TRAIN_MODEL_PATH=./model/trained_model/new_best_val_acc_model.h5

python3 second_train.py \
		--batch_size 10 \
		--epoch 3 \
		--max_length 256 \
		--gelu tanh \
		--train_data_dir ${TRAIN_DATA_PATH} \
		--valid_data_dir ${VALID_DATA_PATH} \
		--test_data_dir ${TEST_DATA_PATH} \
		--vocab_path ${VOCAB_PATH}\
		--trained_model_path ${TRAINED_MODEL_PATH}\
		--second_train_model_path ${SECOND_TRAIN_MODEL_PATH}
