import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class PhaseConv(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # 1x1卷积，线性处理，不加激活
        self.conv1 = nn.Conv2d(in_ch*2, in_ch*2, 1, padding=0)
        self.conv2 = nn.Conv2d(in_ch*2, in_ch, 1, padding=0)  # 压回原通道数，可选

    def forward(self, phase):
        # phase -> cos/sin 映射
        phase_cos = torch.cos(phase)   # [B, C, H, W]
        phase_sin = torch.sin(phase)   # [B, C, H, W]
        phase_feat = torch.cat([phase_cos, phase_sin], dim=1)  # [B, 2*C, H, W]

        # 线性卷积
        out = self.conv1(phase_feat)
        out = self.conv2(out)
        return out
    
class SpatFreqBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 原始特征卷积
        # self.x_conv = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch, 3, padding=1),
        #     #nn.BatchNorm2d(in_ch),
        #     nn.GELU(),
        #     nn.Conv2d(in_ch, in_ch, 3, padding=1),
        #     #nn.BatchNorm2d(in_ch),
        #     nn.GELU(),
        # )
        # FFT amplitude 卷积
        self.amp_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, padding=0),
            #nn.BatchNorm2d(in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, 1, padding=0),
        )
        # self.phase_conv = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch, 1, padding=0),
        #     #nn.BatchNorm2d(in_ch),
        #     nn.GELU(),
        #     nn.Conv2d(in_ch, in_ch, 1, padding=0),
        # )
        self.phase_conv = PhaseConv(in_ch)
        
        #self.spatialfreq_attn = SwinSpatialBlock(dim=in_ch*2, window_size=window_size, heads=heads)
        # self.conv_out = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 1, padding=0),
        #     #nn.BatchNorm2d(out_ch),
        #     nn.GELU(),
        # )
        #print("conv_out:", in_ch*2, "->", out_ch)
        self.amp_conv.to(dtype=torch.float32)
        self.phase_conv.to(dtype=torch.float32)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            #nn.BatchNorm2d(in_ch),
            nn.GELU(),
        )

    def forward(self, x:torch.Tensor):
        # FFT amplitude
        identity = x
        F = torch.fft.fft2(x)         
        F = torch.fft.fftshift(F)
        amp = torch.abs(F)
        phase = torch.angle(F)
        amp = self.amp_conv(amp)
        phase = self.phase_conv(phase)
        F = amp * torch.exp(1j * phase)
        F = torch.fft.ifftshift(F)
        fft_x = torch.fft.ifft2(F).real

        # 原始特征卷积
        #x_conv = self.x_conv(x) + identity
        fft_x = fft_x + identity
        # 拼接原始 + FFT幅值
        #fused = torch.cat([x_conv, fft_x], dim=1)  # [B, 2*in_ch, H, W]
        # 自注意力
        #out, gate_vis = self.spatialfreq_attn(fused)
        out = self.fuse_conv(fft_x) 
        #out = self.conv_out(out)
        return out
    
    
    
class FFT_Refiner(nn.Module):
    def __init__(self, in_ch=6, out_ch=3, 
                 base_chs=[16, 32, 64, 128]):
        super().__init__()

        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_ch, base_chs[0], 3, padding=1),
            #nn.BatchNorm2d(base_chs[0]),
            nn.GELU()
        )
        prev_ch = base_chs[0]
        for ch in base_chs[:-1]:
            self.enc_blocks.append(SpatFreqBlock(prev_ch, ch))
            prev_ch = ch

        # Bottleneck
        self.bottleneck = SpatFreqBlock(prev_ch, base_chs[-1])
        prev_ch = base_chs[-1]

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.skip_conv_11 = nn.ModuleList()
        
        for ch in reversed(base_chs[:-1]):
            self.up_convs.append(nn.ConvTranspose2d(prev_ch, ch, 2, stride=2))
            self.skip_conv_11.append(nn.Conv2d(ch, ch, 1))
            self.up_blocks.append(SpatFreqBlock(prev_ch, ch))
            prev_ch = ch

        # Final Conv
        self.conv_out = nn.Conv2d(prev_ch, out_ch, 1)



    def forward(self, x):
        
        #x = (x - self.mean_proc.to(x.device)) / self.std_proc.to(x.device)
        #print(x.shape)
        with torch.cuda.amp.autocast(enabled=False):
            enc_feats = []
            x = x.to(dtype=torch.float32)
            out = self.conv_in(x)
            # ===== Encoder =====
            for block in self.enc_blocks:
                if self.training:
                    out = checkpoint(block, out,use_reentrant = False)
                else:
                    out = block(out)

                enc_feats.append(out)
                out = self.pool(out)

            # ===== Bottleneck =====
            if self.training:
                out = checkpoint(self.bottleneck, out,use_reentrant = False)
            else:
                out = self.bottleneck(out)

            # ===== Decoder =====
            for up_conv, up_block, skip, skip_conv in zip(
                self.up_convs,
                self.up_blocks,
                reversed(enc_feats),
                self.skip_conv_11
            ):
                out = up_conv(out)

                if out.shape[-2:] != skip.shape[-2:]:
                    diffH = skip.size(2) - out.size(2)
                    diffW = skip.size(3) - out.size(3)
                    out = F.pad(out, [0, diffW, 0, diffH])

                skip = skip_conv(skip)

                out = torch.cat([out, skip], dim=1)

                if self.training:
                    out = checkpoint(up_block, out,use_reentrant = False)
                else:
                    out = up_block(out)

            out = self.conv_out(out)
        return out