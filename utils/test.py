# import numpy as np
# image_specs = np.load("/blob/lzy/AR/Reconstruction/dataset/filenames_2.npy")
# print(f"len(image_specs) {len(image_specs)} {image_specs[0]}")

# import sys
# sys.path.append("/data/code/stable-audio-tools/stable_audio_tools/models")
# from process_mm_info import process_mm_info

# path = "/blob/vggsound/vggsound_03/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/A-ZzvGPcdOI_000011.mp4"
# prompts = [path]
# conversation = []
# for temp in prompts:
#     if type(temp) == str or len(temp) == 1:
#         video_path, video_start, video_end = temp, 0, None
#     else:
#         video_path, video_start, video_end = temp
#     if video_start != 0: print("*******************video_start != 0")
#     conversation.append(
#         {
#             "role": "user",
#             "content": [
#                 {"type": "video", "video": video_path, "video_start": video_start, "video_end": video_end},
#             ],
#         }
#     )
# print("conversation:", conversation)
# audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B", torch_dtype="auto", device_map="auto")
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

# 提取音频编码器
audio_encoder = model.thinker.audio_tower
video_encoder = model.thinker.visual
# 保存为独立权重
audio_encoder.save_pretrained("/blob/lzy/AR/cache/qwen_audio_encoder")

# 以后单独加载
from transformers import AutoModel
audio_encoder2 = AutoModel.from_pretrained("/blob/lzy/AR/cache/qwen_audio_encoder")


# from transformers import Qwen2_5OmniForConditionalGeneration
# import inspect

# # 查看定义文件路径
# print(inspect.getfile(Qwen2_5OmniForConditionalGeneration))

# # 查看 forward 源代码
# print(inspect.getsource(Qwen2_5OmniForConditionalGeneration.forward))
