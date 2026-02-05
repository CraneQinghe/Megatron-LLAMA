from datetime import datetime  
import os
import json
class Comm_counter:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Comm_counter, cls).__new__(cls, *args, **kwargs)
            cls._instance.iteration = 0  # 初始化属性
            cls._instance.sequence_parallel_async_forward_all_gather_total_size_MB=0
            cls._instance.sequence_parallel_async_forward_all_gather_buffer_shape=None
            cls._instance.sequence_parallel_async_backward_all_gather_total_size_MB=0
            cls._instance.sequence_parallel_async_backward_all_gather_buffer_shape=None
            cls._instance.async_grad_allreduce_backward_grad_input_total_size_MB=0
            cls._instance.async_grad_allreduce_backward_grad_input_shape=None
            cls._instance.sequence_parallel_async_backward_reduce_scatter_total_size_MB=0
            cls._instance.sequence_parallel_async_backward_reduce_scatter_grad_input_shape=None
            cls._instance.single_all_gather_buffer_size = 0  
            cls._instance.allreduce_layernorm_grads_total_size_MB=0
            cls._instance.allreduce_layernorm_grads_shape=None
            cls._instance.allreduce_embedding_grads_total_size_MB=0
            cls._instance.allreduce_embedding_grads_shape=None
            cls._instance.reduce_scatter_tensor_total_size_MB=0
            cls._instance.reduce_scatter_tensor_shape=None
            cls._instance.data_parallel_group_rank=0
            cls._instance.all_gather_tensor_total_size_MB=0
            cls._instance.all_gather_tensor_shape=None
        return cls._instance

    def set_iteration(self, iteration):
        self.iteration =iteration

    def add_sequence_parallel_async_forward_all_gather_total_size_MB(self,size_MB):
        self.sequence_parallel_async_forward_all_gather_total_size_MB += size_MB
    
    def set_sequence_parallel_async_forward_all_gather_buffer_shape(self,shape):
        if not self.sequence_parallel_async_forward_all_gather_buffer_shape:
            self.sequence_parallel_async_forward_all_gather_buffer_shape = shape
    
    def add_sequence_parallel_async_backward_all_gather_total_size_MB(self,size_MB):
        self.sequence_parallel_async_backward_all_gather_total_size_MB +=size_MB
    
    def set_sequence_parallel_async_backward_all_gather_buffer_shape(self,shape):
        if not self.sequence_parallel_async_backward_all_gather_buffer_shape:
            self.sequence_parallel_async_backward_all_gather_buffer_shape = shape
    
    def add_async_grad_allreduce_backward_grad_input_total_size_MB(self,size_MB):
        self.async_grad_allreduce_backward_grad_input_total_size_MB +=size_MB
    
    def set_async_grad_allreduce_backward_grad_input_shape(self,shape):
        if not self.async_grad_allreduce_backward_grad_input_shape:
            self.async_grad_allreduce_backward_grad_input_shape = shape
    
    def add_sequence_parallel_async_backward_reduce_scatter_total_size_MB(self,size_MB):
        self.sequence_parallel_async_backward_reduce_scatter_total_size_MB +=size_MB
    
    def set_sequence_parallel_async_backward_reduce_scatter_grad_input_shape(self,shape):
        if not self.sequence_parallel_async_backward_reduce_scatter_grad_input_shape:
            self.sequence_parallel_async_backward_reduce_scatter_grad_input_shape = shape
    
    def set_single_all_gather_buffer_size(self,size_MB):
        if not self.single_all_gather_buffer_size:
            self.single_all_gather_buffer_size=size_MB
    
    def add_allreduce_layernorm_grads_total_size_MB(self,size_MB):
        self.allreduce_layernorm_grads_total_size_MB+=size_MB
    
    def set_allreduce_layernorm_grads_shape(self,shape):
        if not self.allreduce_layernorm_grads_shape:
            self.allreduce_layernorm_grads_shape = shape
    
    def add_allreduce_embedding_grads_total_size_MB(self,size_MB):
        self.allreduce_embedding_grads_total_size_MB+=size_MB
    
    def set_allreduce_embedding_grads_shape(self,shape):
        if not self.allreduce_embedding_grads_shape:
            self.allreduce_embedding_grads_shape = shape
    
    def add_reduce_scatter_tensor_total_size_MB(self,size_MB):
        self.reduce_scatter_tensor_total_size_MB+=size_MB
    
    def set_reduce_scatter_tensor_shape(self,shape):
        if not self.reduce_scatter_tensor_shape:
            self.reduce_scatter_tensor_shape = shape  
    
    def add_all_gather_tensor_total_size_MB(self,size_MB):
        self.all_gather_tensor_total_size_MB+=size_MB
    
    def set_all_gather_tensor_shape(self,shape):
        if not self.all_gather_tensor_shape:
            self.all_gather_tensor_shape = shape  
    def set_data_parallel_group_rank(self,nranks):
        self.data_parallel_group_rank=nranks
    
    def dump(self):
        now = datetime.now()
        date_str=now.strftime("%m-%d")
        time_str=now.strftime("%H-%M")
        folder_path = f"/data/haiqwa/zevin_nfs/code/Megatron-LLaMA/examples/LLaMA/logs/comm_time/{date_str}/{time_str}/"
        os.makedirs(folder_path, exist_ok=True)
        json_file_path = f"{folder_path}/comm_info.json"
        if not os.path.exists(json_file_path):
            # 初始化一个空的 JSON 文件
            with open(json_file_path, "a") as f:
                data = {
                    "Framework":"megatron",
                    "iteration":self.iteration,
                    "single_all_gather_buffer_size":self.single_all_gather_buffer_size,
                    "sequence_parallel_async_forward_all_gather_total_size_MB": self.sequence_parallel_async_forward_all_gather_total_size_MB,
                    "sequence_parallel_async_forward_all_gather_buffer_shape": self.sequence_parallel_async_forward_all_gather_buffer_shape,
                    "sequence_parallel_async_backward_all_gather_total_size_MB": self.sequence_parallel_async_backward_all_gather_total_size_MB,
                    "sequence_parallel_async_backward_all_gather_buffer_shape": self.sequence_parallel_async_backward_all_gather_buffer_shape,
                    "async_grad_allreduce_backward_grad_input_total_size_MB": self.async_grad_allreduce_backward_grad_input_total_size_MB,
                    "async_grad_allreduce_backward_grad_input_shape": self.async_grad_allreduce_backward_grad_input_shape,
                    "sequence_parallel_async_backward_reduce_scatter_total_size_MB": self.sequence_parallel_async_backward_reduce_scatter_total_size_MB,
                    "sequence_parallel_async_backward_reduce_scatter_grad_input_shape": self.sequence_parallel_async_backward_reduce_scatter_grad_input_shape,
                    "allreduce_layernorm_grads_total_size_MB":self.allreduce_layernorm_grads_total_size_MB,
                    "allreduce_layernorm_grads_shape":self.allreduce_layernorm_grads_shape,
                    "allreduce_embedding_grads_total_size_MB":self.allreduce_embedding_grads_total_size_MB,
                    "allreduce_embedding_grads_shape":self.allreduce_embedding_grads_shape,
                    "reduce_scatter_tensor_total_size_MB":self.reduce_scatter_tensor_total_size_MB,
                    "reduce_scatter_tensor_shape":self.reduce_scatter_tensor_shape,
                    "all_gather_tensor_total_size_MB":self.all_gather_tensor_total_size_MB,
                    "all_gather_tensor_shape":self.all_gather_tensor_shape,
                    "data_parallel_group_rank":self.data_parallel_group_rank
                    }
                json.dump(data, f, indent=4)
