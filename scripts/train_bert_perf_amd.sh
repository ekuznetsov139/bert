export HIP_VISIBLE_DEVICES=0 # choose gpu

CODE_DIR=.
DATA_DIR=./data
TRAIN_DIR=./bert_perf_train
MODEL_CONFIG_DIR=configs/bert_base

rm -rf $TRAIN_DIR
mkdir -p $TRAIN_DIR
mkdir -p $DATA_DIR

# prep train dir
cp $MODEL_CONFIG_DIR/vocab.txt $TRAIN_DIR/vocab.txt
cp $MODEL_CONFIG_DIR/bert_config.json $TRAIN_DIR/bert_config.json

DATA_SOURCE_FILE_PATH=data/wiki_00
DATA_SOURCE_NAME=$(basename "$DATA_SOURCE_FILE_PATH")

# iterate through configs (Batch, Sequence Length)
for CONFIG in 20,128; do

  IFS=","
  set -- $CONFIG

  BATCH=$1
  SEQ=$2

  CUR_TRAIN_DIR=$TRAIN_DIR/${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}
  mkdir -p $CUR_TRAIN_DIR

  DATA_TFRECORD=$DATA_DIR/${DATA_SOURCE_NAME}_seq${SEQ}.tfrecord
  if [ ! -f "$DATA_TFRECORD" ]; then
    # generate tfrecord of data
    python3 $CODE_DIR/create_pretraining_data.py \
      --input_file=$CODE_DIR/$DATA_SOURCE_FILE_PATH \
      --output_file=$DATA_TFRECORD \
      --vocab_file=$TRAIN_DIR/vocab.txt \
      --do_lower_case=True \
      --max_seq_length=$SEQ \
      --max_predictions_per_seq=20 \
      --masked_lm_prob=0.15 \
      --random_seed=12345 \
      --dupe_factor=5
  fi

  # run pretraining
  python3 $CODE_DIR/run_pretraining.py \
   --input_file=$DATA_TFRECORD \
    --output_dir=$CUR_TRAIN_DIR \
    --do_train=True \
    --do_eval=False \
    --bert_config_file=$TRAIN_DIR/bert_config.json \
    --train_batch_size=$BATCH \
    --max_seq_length=$SEQ \
    --max_predictions_per_seq=20 \
    --num_train_steps=100 \
    --num_warmup_steps=50 \
    --learning_rate=1e-6 \
    --use_horovod=False \
    --use_fp16=True \
    --use_xla=1 \
    2>&1 | tee $CUR_TRAIN_DIR/${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}.txt

done