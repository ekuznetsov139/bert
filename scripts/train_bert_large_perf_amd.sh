# export CUDA_VISIBLE_DEVICES=0 # choose gpu
export HIP_VISIBLE_DEVICES=0 # choose gpu

CODE_DIR=.
DATA_DIR=./bert_perf_data
TRAIN_DIR=./bert_large_perf_train
MODEL_CONFIG_DIR=configs/bert_large

rm -rf $TRAIN_DIR
mkdir -p $TRAIN_DIR
mkdir -p $DATA_DIR

# prep train dir
cp $MODEL_CONFIG_DIR/vocab.txt $TRAIN_DIR/vocab.txt
cp $MODEL_CONFIG_DIR/bert_config.json $TRAIN_DIR/bert_config.json

DATA_SOURCE_FILE_PATH=data/train_en.txt
DATA_SOURCE_FILE=$(basename "$DATA_SOURCE_FILE_PATH")
DATA_SOURCE_NAME="${DATA_SOURCE_FILE%.*}"

# iterate through configs (possible values 64,64 32,128 16,256 6,512)
for CONFIG in 4,512; do
  IFS=","
  set -- $CONFIG

  CUR_TRAIN_DIR=$TRAIN_DIR/${DATA_SOURCE_NAME}_ba${1}_seq${2}
  mkdir -p $CUR_TRAIN_DIR

  # generate tfrecord of data
  python3 $CODE_DIR/create_pretraining_data.py \
    --input_file=$CODE_DIR/$DATA_SOURCE_FILE_PATH \
    --output_file=$DATA_DIR/${DATA_SOURCE_NAME}_seq${2}.tfrecord \
    --vocab_file=$TRAIN_DIR/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=$2 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5

  # run pretraining
  python3 $CODE_DIR/run_pretraining.py \
    --input_file=$DATA_DIR/${DATA_SOURCE_NAME}_seq${2}.tfrecord \
    --output_dir=$CUR_TRAIN_DIR \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=$TRAIN_DIR/bert_config.json \
    --train_batch_size=$1 \
    --max_seq_length=$2 \
    --max_predictions_per_seq=20 \
    --num_train_steps=1500 \
    --num_warmup_steps=150 \
    --learning_rate=2e-5 \
    2>&1 | tee $CUR_TRAIN_DIR/${DATA_SOURCE_NAME}_ba${1}_seq${2}.txt

done
