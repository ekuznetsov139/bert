export FP16=1

export TF_ROCM_GELU=0
export TF_BIAS_GRAD_MODE=1
export TF_BIAS_DIV=1
export TF_ROCM_OLD_DROPOUT=1
export TF_ROCM_FMA_DISABLE=1
export XLA=0

sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0

export TF_ROCM_GELU=1
export TF_BIAS_GRAD_MODE=1
export TF_BIAS_DIV=1
export TF_ROCM_OLD_DROPOUT=0
export TF_ROCM_FMA_DISABLE=0
export XLA=0

sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0

export TF_ROCM_GELU=1
export TF_BIAS_GRAD_MODE=1
export TF_BIAS_DIV=1
export TF_ROCM_OLD_DROPOUT=0
export TF_ROCM_FMA_DISABLE=0
export XLA=1

sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0

export TF_ROCM_GELU=0
export TF_BIAS_GRAD_MODE=1
export TF_BIAS_DIV=1
export TF_ROCM_OLD_DROPOUT=1
export TF_ROCM_FMA_DISABLE=1
export XLA=1

sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0
sh scripts//train_bert_large_perf_amd.sh 0

