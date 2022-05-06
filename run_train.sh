#ROOT_DIR=""
BERT_PRETRAINED_DIR="facebook/xglm-564M"
CHECKPOINT_DIR="."
DATA_PREFIX="./data"

#source ${ROOT_DIR}/.bashrc

#export CUDA_VISIBLE_DEVICES=1
python train.py \
--model_name CSNZeroshot \
--pooling_type max_pooling \
--dropout 0.5 \
--optimizer adamw \
--margin 1.0 \
--lr 5e-5 \
--num_epochs 1 \
--batch_size 16 \
--patience 10 \
--bert_pretrained_dir ${BERT_PRETRAINED_DIR} \
--train_file \
${DATA_PREFIX}/train/train_unsplit.txt \
--dev_file \
${DATA_PREFIX}/dev/dev_unsplit.txt \
--test_file \
${DATA_PREFIX}/test/test_unsplit.txt \
--name_list_path \
${DATA_PREFIX}/name_list.txt \
--length_limit 510 \
--checkpoint_dir ${CHECKPOINT_DIR} \
--num_shots 50 \
--prompt_lang zh \
--score_mode lm_probs #\
#--use_full_context