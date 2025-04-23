"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
import math


#Local imports
from model.modules import BaseRGBModel, FCLayers, step
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def freeze_temporal_stride(module):
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv3d):
            if isinstance(m.stride, tuple) and m.stride[0] > 1:
                m.stride = (1, m.stride[1], m.stride[2])
    return module

class TCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, num_layers=3, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(nn.Conv1d(in_channels if i == 0 else hidden_dim,
                                    hidden_dim, kernel_size,
                                    padding=padding, dilation=dilation))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
        self.out_proj = nn.Conv1d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, x):  # x: [B, T, C]
        x = x.permute(0, 2, 1)      # [B, C, T]
        out = self.network(x)       # [B, H, T]
        out = self.out_proj(out)    # [B, C, T]
        return out.permute(0, 2, 1) # [B, T, C]


class SGPModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.inst_fc1 = nn.Linear(in_channels, in_channels // 2)
        self.inst_fc2 = nn.Linear(in_channels // 2, in_channels)

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.fuse = nn.Conv1d(in_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):  # x: [B, T, C]
        B, T, C = x.shape

        x_mean = x.mean(dim=1, keepdim=True)
        x_inst = x - x_mean
        x_inst = F.relu(self.inst_fc1(x_inst))
        x_inst = self.inst_fc2(x_inst)

        x_window = x.permute(0, 2, 1)  # [B, C, T]
        x1 = self.conv1(x_window)
        x2 = self.conv2(x_window)
        x3 = self.conv3(x_window)
        x_cat = torch.cat([x1, x2, x3], dim=1)  # [B, 3C, T]
        x_fused = self.fuse(x_cat)  # [B, C, T]
        x_fused = x_fused.permute(0, 2, 1)  # [B, T, C]

        return x_inst + x_fused


class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            if self._feature_arch == 'x3d_s':
                x3d_full = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
                x3d_full = freeze_temporal_stride(x3d_full) 
                features = nn.Sequential(*x3d_full.blocks[:5])
                self._d = 192  # output block 5
            else:
                raise NotImplementedError(f"Unsupported feature arch: {self._feature_arch}")

            self._features = features

            self._tcn = TCNBlock(in_channels=self._d)
            self._sgp = SGPModule(in_channels=self._d)

            # MLP for classification
            self._fc = FCLayers(self._d, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

        def forward(self, x):
            x = self.normalize(x)
            batch_size, clip_len, channels, height, width = x.shape

            if self.training:
                x = self.augment(x)

            x = self.standarize(x)
            
            x = x.permute(0, 2, 1, 3, 4)
            x = self._features(x) # [B, C, T, H, W]
            x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))   # [B, C, T, 1, 1]
            x = x.squeeze(-1).squeeze(-1)  # [B, C, T]
            x = x.permute(0, 2, 1) # [B, T, C]
        
            # TCN
            x = self._tcn(x)  # [B, T, C]

            # SGP
            x = self._sgp(x)  # [B, T, C]

            # MLP
            im_feat = self._fc(x) # [B, T, num_classes + 1]

            return im_feat 
        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
    
            dummy_input_0 = torch.randn(1, 1, 3, 224, 224).to(next(self.parameters()).device) 
            dummy_input = torch.randn(4, 50, 3, 224, 398).to(next(self.parameters()).device) 
    
            self.eval()
            try:
                flops_0 = FlopCountAnalysis(self, dummy_input_0)
                flops = FlopCountAnalysis(self, dummy_input)
                print(f"FLOPs_0: {flops_0.total():,}")
                print(f"FLOPs: {flops.total():,}")
                print(parameter_count_table(self))
            except Exception as e:
                print(f"Could not calculate FLOPs: {e}")

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes
                    label = label.view(-1) # B*T
                    loss = F.cross_entropy(
                            pred, label, reduction='mean', weight = weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.softmax(pred, dim=-1)
            
            return pred.cpu().numpy()
