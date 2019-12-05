if [ $# -eq 0 ]; then
    SEQ=512
else
    SEQ=$1
fi

DATA_DIR=./data/wikipedia
WIKI_TEXT_DIR=${DATA_DIR}/wiki_text
WIKI_TFRECORD_DIR=$DATA_DIR/wiki_tfrecord_seq${SEQ}

mkdir -p $WIKI_TFRECORD_DIR

# generate tfrecord of data in parallel
for DIR in ${WIKI_TEXT_DIR}/*; do
    for FILE in $DIR/*; do
        DIR_BASENAME=$(basename $DIR)
        FILE_BASENAME=$(basename $FILE)
        python3 create_pretraining_data.py \
            --input_file=${FILE} \
            --output_file=$WIKI_TFRECORD_DIR/${DIR_BASENAME}--${FILE_BASENAME}.tfrecord \
            --vocab_file=configs/bert_large/vocab.txt \
            --do_lower_case=True \
            --max_seq_length=${SEQ} \
            --max_predictions_per_seq=20 \
            --masked_lm_prob=0.15 \
            --random_seed=12345 \
            --dupe_factor=5 &
        sleep 1
    done
done

wait
echo "Done preprocessing wikipedia"