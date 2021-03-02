SCRIPTPATH=$(dirname $(realpath $0))
# export HIP_VISIBLE_DEVICES=0 # choose gpu

CODE_DIR=/root/bert
DATA_DIR=/data/wikipedia/eugene
TRAIN_DIR=./bert_full_train
MODEL_CONFIG_DIR=configs/bert_large

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=4

#rm -rf $TRAIN_DIR
#mkdir -p $TRAIN_DIR
#rm -rf $TRAIN_DIR/*
mkdir -p $DATA_DIR

# prep train dir
cp $MODEL_CONFIG_DIR/vocab.txt $TRAIN_DIR/vocab.txt
cp $MODEL_CONFIG_DIR/bert_config.json $TRAIN_DIR/bert_config.json

# iterate through configs (Batch, Sequence Length)
for CONFIG in 10,128; do

  IFS=","
  set -- $CONFIG

  BATCH=$1
  SEQ=$2

  CUR_TRAIN_DIR=$TRAIN_DIR/ba${BATCH}_seq${SEQ}
  mkdir -p $CUR_TRAIN_DIR

  WIKI_TFRECORD_DIR=$DATA_DIR/wiki_tfrecord_seq${SEQ}
  if [ ! -d "$WIKI_TFRECORD_DIR" ]; then
    sh $SCRIPTPATH/preprocess_wikipedia.sh ${SEQ}
  fi

  # run pretraining
  horovodrun --verbose -np 8 -H localhost:8 python3 $CODE_DIR/run_pretraining.py \
    --input_file=$WIKI_TFRECORD_DIR/*.tfrecord \
    --output_dir=$CUR_TRAIN_DIR \
    --do_train=True \
    --do_eval=False \
    --bert_config_file=$TRAIN_DIR/bert_config.json \
    --train_batch_size=$BATCH \
    --max_seq_length=$SEQ \
    --max_predictions_per_seq=20 \
    --num_train_steps=1000000 \
    --num_warmup_steps=100000 \
    --learning_rate=1e-4 \
    --use_horovod=True \
    --use_fp16=True \
    --use_xla=1 \
    |& tee -a train128.txt

  parallel -j 8 < scripts/eval_commands
  python3 scripts/parse_eval.py
done
