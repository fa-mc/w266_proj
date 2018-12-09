# Chinese Sentiment Analysis with Google BERT
### Instructions

The Bert folder is cloned from Google's official repository (git pull git@github.com:google-research/bert.git). I have edited the run_classifier.py file to make it compatible with our processed ChnSentiCorp data.


#### 1. Download and extract the BERT base Chinese monolanguage model into ./bert/models
```
mkdir $PROJ/bert/models
cd $PROJ/bert/models
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip
```

#### 2. (skip 2 & 3 if you want to use the processed data) unzip the ChnSentiCorp data
```
cd $PROJ/data/raw/
unzip chnsenticorp.zip ./data/raw/
```

#### 3. (skip 2 & 3 if you want to use the processed data) Run process_csc.ipynb to process the raw files

#### 4. Run the following command to train and evaluate the models
The official document from Google used python2 and TensorFlow 1.11 I ran the code with python3.6 and TensorFlow 1.10 without error.

Increasing the max_seq_length and / or train_batch_size might yield better results. My PC (GTX 1070) is limited to 128 / 20.

```
cd $PROJ/bert
# dangdang
python run_classifier.py --task_name csc --do_train --do_eval \
--data_dir $PROJ/data/processed/csc/dangdang \
--vocab_file $PROJ/bert/models/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file $PROJ/bert/models/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint $PROJ/bert/models/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length 128 --train_batch_size 20 --learning_rate 2e-5 --num_train_epochs 3.0 \
--output_dir $PROJ/output/dangdang --local_rank 3

# ctrip
python run_classifier.py --task_name csc --do_train --do_eval \
--data_dir $PROJ/data/processed/csc/ctrip \
--vocab_file $PROJ/bert/models/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file $PROJ/bert/models/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint $PROJ/bert/models/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length 128 --train_batch_size 20 --learning_rate 2e-5 --num_train_epochs 3.0 \
--output_dir $PROJ/output/ctrip --local_rank 3

# jingdong
python run_classifier.py --task_name csc --do_train --do_eval \
--data_dir $PROJ/data/processed/csc/jingdong \
--vocab_file $PROJ/bert/models/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file $PROJ/bert/models/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint $PROJ/bert/models/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length 128 --train_batch_size 20 --learning_rate 2e-5 --num_train_epochs 3.0 \
--output_dir $PROJ/output/jingdong --local_rank 3

# combined
python run_classifier.py --task_name csc --do_train --do_eval \
--data_dir $PROJ/data/processed/csc/all \
--vocab_file $PROJ/bert/models/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file $PROJ/bert/models/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint $PROJ/bert/models/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length 128 --train_batch_size 20 --learning_rate 2e-5 --num_train_epochs 3.0 \
--output_dir $PROJ/output/csc_all --local_rank 3
```
