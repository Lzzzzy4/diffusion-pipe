from videoconditioner import VideoEncoderConditioner

model = VideoEncoderConditioner(output_dim=768)
# path = "/blob/vggsound/vggsound_03/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/A-ZzvGPcdOI_000011.mp4"
# path1 = "/blob/vggsound/vggsound_14/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/gm-U_oCATbE_000006.mp4"
# path2 = "/blob/vggsound/vggsound_03/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/8OHjYLy7AoA_000070.mp4"
path1 = "/blob/vggsound/vggsound_08/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/OQXvSxcAjwM_000053.mp4"
# path2 = "/blob/vggsound/vggsound_08/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/MeanyeFs_4o_000030.mp4"
output1 = model([path1], device="cuda", demo=True)
# output2 = model([path], device="cuda", demo=True)
# output3 = model([path1], device="cuda", demo=True)
# output4 = model([path2], device="cuda", demo=True)

# import numpy as np
# # check if output1 is the same as [output2, output3, output4]
# import torch


# new_output1 = torch.cat([output2, output3, output4], dim=0)
# # check if eq
# print(torch.allclose(output1, new_output1, atol=1e-5))
