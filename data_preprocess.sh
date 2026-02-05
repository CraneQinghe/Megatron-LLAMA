python tools/preprocess_data.py \
       --input /data/haiqwa/zevin_nfs/dataset/bookcorpus_llama3_4K/bookcorpus.json \
       --output-prefix /data/haiqwa/zevin_nfs/code/Megatron-LLaMA/examples/LLaMA/dataset \
       --tokenizer-name-or-path /data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/qinghe/nnscaler-main/examples/llama3_8B_128K/llama3_mini \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-model /data/haiqwa/gpt/jcz/Meta-Llama-3.1-8B-Instruct \
       --workers 16 \
       --chunk-size 32 \
       --append-eod