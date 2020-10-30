
# Preliminary
Excute below scripts in main folder firstly.
```
cd pytorch_pretrained_bert/bert-base-uncased/
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
mv bert-base-uncased-vocab.txt vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
tar -xvf bert-base-uncased.tar.gz
rm bert-base-uncased.tar.gz
cd ../../
```

# Finetune on YoucookII
## Retrieval
```
INIT_MODEL=<from second phase>
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_retrieval_task.py \
--do_train --num_thread_reader=16 \
--epochs=5 --batch_size=32 \
--n_pair=1 --n_display=100 \ 
--youcook_train_csv data/youcookii_train.csv \
--youcook_val_csv data/youcookii_val.csv \
--youcook_caption_path data/youcookii_caption.pickle \
--youcook_features_path_2D data/youcookii_videos_features \
--output_dir ckpt_youcook_retrieval --bert_model bert-base-uncased 、
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 200 --visual_num_hidden_layers 6 \
--datatype youcook --init_model ${INIT_MODEL}
```

## Caption
```
INIT_MODEL=<from second phase>
python -m torch.distributed.launch --nproc_per_node=4 train_transcript_distributed.py \
--do_train --num_thread_reader=4 \
--epochs=5 --batch_size=16 \
--n_pair=-1 --n_display=100 \
--youcook_train_csv data/youcookii_train.csv \
--youcook_val_csv data/youcookii_val.csv \
--youcook_caption_path data/youcookii_caption_transcript.pickle \
--youcook_features_path_2D data/youcookii_videos_features \
--output_dir ckpt_youcook_caption --bert_model bert-base-uncased \
--do_lower_case --lr 3e-5 --max_words 128 --max_frames 96 \
--batch_size_val 64 --visual_num_hidden_layers 6 \
--decoder_num_hidden_layers 3 \
--init_model ${INIT_MODEL}
```

# Pretrain on HowTo100M
## Phase I
```
ROOT_PATH=.
DATA_PATH=${ROOT_PATH}/data
SAVE_PATH=${ROOT_PATH}/models
MODEL_PATH=${ROOT_PATH}/UniVL
python -m torch.distributed.launch --nproc_per_node=8 ${MODEL_PATH}/train_transcript_distributed.py \
 --do_pretrain --num_thread_reader=0 --epochs=50 \
--batch_size=1920 --n_pair=3 --n_display=100 \
--bert_model bert-base-uncased --do_lower_case --lr 1e-4 \
--max_words 48 --max_frames 64 --batch_size_val 344 \
--output_dir ${SAVE_PATH}/pre_trained/pre_s3d_L48_V6_D3_Phase1 \
--features_path_2D ${DATA_PATH}/features \
--train_csv ${DATA_PATH}/HowTo100M.csv \
--caption_path ${DATA_PATH}/caption.pickle \
--visual_num_hidden_layers 6 --gradient_accumulation_steps 16 \
--sampled_use_mil --load_checkpoint
```

## Phase II
```
ROOT_PATH=.
DATA_PATH=${ROOT_PATH}/data
SAVE_PATH=${ROOT_PATH}/models
MODEL_PATH=${ROOT_PATH}/UniVL
INIT_MODEL=<from first phase>
python -m torch.distributed.launch --nproc_per_node=8 ${MODEL_PATH}/train_transcript_distributed.py \
--do_pretrain --num_thread_reader=0 --epochs=50 \
--batch_size=960 --n_pair=3 --n_display=100 \
--bert_model bert-base-uncased --do_lower_case --lr 1e-4 \
--max_words 48 --max_frames 64 --batch_size_val 344 \
--output_dir ${SAVE_PATH}/pre_trained/pre_s3d_L48_V6_D3_Phase2 \
--features_path_2D ${DATA_PATH}/features \
--train_csv ${DATA_PATH}/HowTo100M.csv \
--caption_path ${DATA_PATH}/caption.pickle \
--visual_num_hidden_layers 6 --decoder_num_hidden_layers 3 \
--gradient_accumulation_steps 60 \
--cross_model --pretrain_with_joint_sim --sampled_use_mil \
--pretrain_enhance_vmodal --pretrain_without_decoder \
--load_checkpoint --init_model ${INIT_MODEL}
```
