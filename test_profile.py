with open("/Users/wangqinghe/Desktop/Hops论文/Megatron-LLAMA/megatron/profiler.py") as f:
    text = f.read()
import re
match = re.search(r"def dump\(self\):", text)
print(text[:match.start()].count("\n"))
