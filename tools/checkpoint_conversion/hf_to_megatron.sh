python tools/checkpoint_conversion/llama_checkpoint_conversion.py \
--load_path "/data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/qinghe/nnscaler-main/examples/llama3_8B_128K/llama3_mini" \
--save_path "/data/haiqwa/zevin_nfs/code/Megatron-LLaMA/test/llama_mini" \
--target_tensor_model_parallel_size 2 \
--target_pipeline_model_parallel_size 1 \
--target_data_parallel_size 4 \
--target_params_dtype "fp16" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
--megatron-path "/data/haiqwa/zevin_nfs/code/moe_queue/Megatron-LM-core_r0.9.0"