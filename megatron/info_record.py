import threading
from datetime import datetime
import os

class Logger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # Ensure that only one instance is created (Singleton pattern)
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check locking pattern
                    cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, base_path='/data/haiqwa/zevin_nfs/code/Megatron-LLaMA/examples/LLaMA/logs/record_info/', file_prefix='log'):
        if not hasattr(self, 'file_name'):  # Prevent reinitialization of the same instance
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.file_name = os.path.join(base_path, f'{file_prefix}_{timestamp}.txt')

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls(*args, **kwargs)  # Calls __new__ and then __init__
        return cls._instance

    def log(self, data):
        with open(self.file_name, 'a') as file:
            file.write(f"{data}\n")


    
    # 在其他文件中使用时也应通过`get_instance`方法来获取同一个实例
    # another_logger = Logger.get_instance()
    # another_logger.log('Another log entry from different file.')