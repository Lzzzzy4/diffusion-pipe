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
sys.path.append("/data/code/AR/VQ_tok")
from ibq import VQConvProjector
import torch
import torch.nn as nn
import typing as tp
import typing as tp
from typing import Any, Callable, Optional, Union

class VideoEncoderConditioner(nn.Module):

    def __init__(
            self,
            output_dim: int,
            enable_connecter_gard: bool = True,
            vq_quant: bool = True,
            return_sqlens: bool = True,
    ):
        super().__init__()
        self.input_dim = 128
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

        self.connector_in_dim = self.input_dim
        self.connector_out_dim = output_dim
        norm = RMSNorm(self.connector_out_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.fill_(math.sqrt(5.5))
        self.connector = nn.Sequential(
            nn.Linear(self.connector_in_dim, self.connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.connector_out_dim, self.connector_out_dim),
            norm,
        ).train(enable_connecter_gard).requires_grad_(enable_connecter_gard)
        
        self.vq_quant = vq_quant
        if vq_quant:
            print("*********WaveEncoderConditioner using VQ quantization")
            self.ibq_projection = VQConvProjector(
                z_channels=self.connector_out_dim,    # 768
                codebook_size=16384,  # codebook size: 16384
                codebook_dim=self.connector_out_dim,  # 768
                use_transformer=False,
                # config=copy.deepcopy(config),  # use the same config as the model
                recon=False,     # whether to use the recon loss
            )

    def forward(self, prompts: tp.List[str], device: tp.Union[torch.device, str], demo: bool=False) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.visual.to(device)
        self.connector.to(device)
        self.ibq_projection.to(device)
        conversation = []
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

        video_features = self.get_video_features(
            pixel_values_videos=inputs["pixel_values_videos"].to(device),
            video_grid_thw=inputs.get("video_grid_thw", None),
        )
        print(f"video_features {video_features.shape}")
        # N \times [seq, 1280] -> embeddings [batch, 2000, 1280]  attention_mask [batch, 2000]
        embeddings = torch.zeros((len(prompts), 2000, self.input_dim), device=device)
        attention_mask = torch.zeros((len(prompts), 2000), device=device, dtype=torch.long)
        for i,video_feature in enumerate(video_features):
            embeddings[i, :video_feature.shape[0], :] = video_feature
            attention_mask[i, :video_feature.shape[0]] = 1
        # print(f"embeddings {embeddings[0][0]}")
        
        # embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.connector(embeddings)
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

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
            # print(f"embeddings_quant {embeddings_quant.shape, embeddings_quant[0][:10]}, cu_seqlens {cu_seqlens}, valid_lengths {valid_lengths}")
            quant_code, code_idx, vq_loss = self.ibq_projection(
                # embeddings,
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
            out_dtype = next(self.connector.parameters()).dtype
            quant_code = quant_code.to(out_dtype)
            return quant_code, attention_mask

        out_dtype = next(self.connector.parameters()).dtype
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