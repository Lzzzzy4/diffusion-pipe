from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import sys
sys.path.append("/data/code/stable-audio-tools/stable_audio_tools/models")
from process_mm_info_ import process_mm_info
import math
from diffusers.models.normalization import RMSNorm
import ffmpeg
import io
import random
import time
import sys
import gc
sys.path.append("/data/code/AR/VQ_tok")
from ibq import VQConvProjector
import torch
import torch.nn as nn
import typing as tp
import typing as tp
from typing import Any, Callable, Optional, Union

MAX_LEN = 1800
# @use_kernel_forward_from_hub("RMSNorm")
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen2_5OmniPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x

class VideoEncoderConditioner(nn.Module):

    def __init__(
            self,
            output_dim: int,
            enable_connecter_gard: bool = True,
            vq_quant: bool = True,
            return_sqlens: bool = True,
    ):
        # 下游接收 512 * 4096
        super().__init__()
        self.input_dim = 2048
        self.output_dim = output_dim
        self.return_sqlens = return_sqlens
        self.enable_connecter_gard = enable_connecter_gard
        # random sleep for 0～5s
        time.sleep(random.randint(0,5))
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

        model_temp = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B", torch_dtype="auto", device_map="cpu")
        self.visual = model_temp.thinker.visual
        self.visual.train(False).requires_grad_(False)
        del model_temp
        gc.collect()
        torch.cuda.empty_cache()

        self.merger = Qwen2_5OmniPatchMerger(dim=self.output_dim, context_dim=self.input_dim, spatial_merge_size=2).requires_grad_(True)
        
        self.vq_quant = vq_quant
        if vq_quant:
            print("*********VideoEncoderConditioner using VQ quantization")
            self.ibq_projection = VQConvProjector(
                z_channels=self.output_dim,    # 768
                codebook_size=16384,  # codebook size: 16384
                codebook_dim=self.output_dim,  # 768
                use_transformer=False,
                # config=copy.deepcopy(config),  # use the same config as the model
                recon=False,     # whether to use the recon loss
            )

    def forward(self, prompts: tp.List[str], device: tp.Union[torch.device, str], demo: bool=False) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.visual.to(device)
        self.merger.to(device)
        self.ibq_projection.to(device)
        conversation = []
        N = len(prompts)
        for temp in prompts:
            if type(temp) is str:
                video_path = temp
                video_start = 0
                video_end = None
            else:
                video_path, video_start, video_end = temp
            if video_start != 0: print("*******************video_start != 0")
            if video_path.split(".")[-1] == 'mp4':
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_path, "video_start": video_start, "video_end": video_end},
                        ],
                    }
                )
            else:
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_path, "video_start": video_start, "video_end": video_end},
                        ],
                    }
                )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        # /home/yifanyang/miniconda3/envs/sao/lib/python3.10/site-packages/qwen_omni_utils/v2_5/audio_process.py
        inputs = self.processor(text="Hello", audio=None, images=None, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        video_features = self.get_video_features( # [multi(N, 3), 2048]
            pixel_values_videos=inputs["pixel_values_videos"].to(device),
            video_grid_thw=inputs.get("video_grid_thw", None),
        )
        video_grid = inputs["video_grid_thw"]  # [N, 3]
        grid_mul = video_grid[:, 0] * video_grid[:, 1] * video_grid[:, 2]
        grid_mul = grid_mul.cpu().numpy().astype(int)  # [N]
        feature_list = []
        start = 0
        for i in range(len(grid_mul)):
            if grid_mul[i] % 4 != 0: raise ValueError(f"grid_mul {grid_mul[i]} not divisible by 4")
            feature_list.append(video_features[start:start+grid_mul[i]//4, :])
            start += grid_mul[i]//4
        for i in range(len(grid_mul)):
            if grid_mul[i]%16 != 0:
                # print(f"**********Warning: grid_mul {grid_mul[i]} not divisible by 16, truncate to {grid_mul[i]-grid_mul[i]%16}")
                remainder = grid_mul[i]%16
                grid_mul[i] = grid_mul[i] - remainder
                feature_list[i] = feature_list[i][:-remainder//4, :]
        video_features = torch.cat(feature_list, dim=0)  # [sum(grid_mul//4), 2048]

        video_features = self.merger(video_features)
        grid_mul = grid_mul / 16  # [N]
        video_features_list = torch.split(video_features, [math.ceil(x) for x in grid_mul], dim=0)
        # <1800 pad 0
        embeddings = torch.zeros((N, MAX_LEN, self.output_dim), device=device)
        attention_mask = torch.zeros((N, MAX_LEN), dtype=torch.long, device=device)
        for i in range(N):
            if video_features_list[i].shape[0] > MAX_LEN:
                # print(f"**********Warning: video_features_list[{i}] length {video_features_list[i].shape[0]} > {MAX_LEN}, truncate to {MAX_LEN}")
                embeddings[i, :, :] = video_features_list[i][:MAX_LEN, :]
                attention_mask[i, :] = 1
            else:
                embeddings[i, :video_features_list[i].shape[0], :] = video_features_list[i]
                attention_mask[i, :video_features_list[i].shape[0]] = 1

        # embeddings = video_features.view(N, -1, self.output_dim)  # [batch, seqlen, dim]
        # attention_mask = torch.ones((N, embeddings.shape[1]), dtype=torch.long, device=device)  # [batch, seqlen=1800]

        if self.vq_quant:
            valid_lengths = attention_mask.sum(dim=1).long()  # [batch]
            cu_seqlens = torch.cat([
                torch.zeros(1, device=device, dtype=torch.long), #[0]
                valid_lengths.cumsum(dim=0) # [batch]
            ], dim=0)  # [batch + 1]
            # embeddings_quant should be [sum(valid_lengths), dim]
            embeddings_quant = torch.zeros((cu_seqlens[-1], embeddings.shape[2]), device=device)
            for i in range(embeddings.shape[0]):
                embeddings_quant[cu_seqlens[i]:cu_seqlens[i+1], :] = embeddings[i, :valid_lengths[i], :]
            quant_code, code_idx, vq_loss = self.ibq_projection(
                embeddings_quant,
                cu_seqlens=cu_seqlens,
                position_embeddings=None,
                demo=demo,
            )
            # back to [batch, seqlen, dim]
            # print(f"quant_code before reshape {quant_code.shape, quant_code[0][:10]}")
            quant_code_batch = torch.zeros_like(embeddings)
            for i in range(embeddings.shape[0]):
                quant_code_batch[i, :valid_lengths[i], :] = quant_code[cu_seqlens[i]:cu_seqlens[i+1], :]
            quant_code = quant_code_batch
            # print(f"quant_code {quant_code.shape, quant_code[0][0][:10]}, code_idx {code_idx.shape}, vq_loss {vq_loss}")
            out_dtype = next(self.merger.parameters()).dtype
            quant_code = quant_code.to(out_dtype)
            return quant_code, valid_lengths

        out_dtype = next(self.merger.parameters()).dtype
        embeddings = embeddings.to(out_dtype)

        return embeddings, attention_mask

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        return video_embeds