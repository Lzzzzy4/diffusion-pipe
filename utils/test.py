from transformers import Qwen2_5OmniForConditionalGeneration
import torch, gc

# 1️⃣ 加载全模型
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B", torch_dtype="auto", device_map="auto"
)


audio_encoder = model.thinker.audio_tower

del model
gc.collect()
torch.cuda.empty_cache()