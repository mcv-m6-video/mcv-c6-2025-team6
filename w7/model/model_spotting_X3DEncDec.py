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

def get_encoder_x3d():
    x3d_full = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
    encoder_blocks = nn.Sequential(*x3d_full.blocks[:-1])
    class TemporalMaxPool(nn.Module):
        def __init__(self, kernel_size):
            super().__init__()
            self.pool = nn.MaxPool3d(kernel_size=(kernel_size, 1, 1), stride=(kernel_size, 1, 1))
        def forward(self, x):
            return self.pool(x)
    temporal_kernel_size = 8
    temporal_pooling = TemporalMaxPool(temporal_kernel_size)
    encoder = nn.Sequential(encoder_blocks, temporal_pooling)
    d_enc = 192
    return encoder, d_enc

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, target_temporal_dim):
        super().__init__()
        self.conv_transpose1 = nn.ConvTranspose3d(in_channels, 128, kernel_size=(5, 1, 1), stride=(2, 1, 1), padding=(2, 0, 0)) # 6 -> 11
        self.bn1 = nn.BatchNorm3d(128)
        self.relu1 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose3d(128, 64, kernel_size=(5, 1, 1), stride=(2, 1, 1), padding=(2, 0, 0)) # 11 -> 21
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.conv_transpose3 = nn.ConvTranspose3d(64, 32, kernel_size=(5, 1, 1), stride=(2, 1, 1), padding=(2, 0, 0)) # 21 -> 41
        self.bn3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU()
        self.final_conv = nn.Conv3d(32, in_channels, kernel_size=(1, 1, 1), stride=1) # Ajusta el número de canales
        self.target_temporal_dim = target_temporal_dim

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv_transpose1(x)))
        x = self.relu2(self.bn2(self.conv_transpose2(x)))
        x = self.relu3(self.bn3(self.conv_transpose3(x)))
        x = self.final_conv(x)
        # Interpolate para alcanzar la dimensión temporal objetivo
        current_temporal_dim = x.shape[2]
        if current_temporal_dim != self.target_temporal_dim:
            x = F.interpolate(x, size=(self.target_temporal_dim, x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=False)
        return x

def get_decoder_x3d(decoder_in_channels):
    return DecoderBlock(decoder_in_channels, 50), decoder_in_channels #TODO: use clip_len instead of fixed 50

class FCLayers(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.fc(x)

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            if self._feature_arch == 'x3d_s':
                self._encoder, self._d_enc = get_encoder_x3d()
                self._decoder, self._d_dec = get_decoder_x3d(decoder_in_channels=self._d_enc)

            else:
                raise NotImplementedError(f"Unsupported feature arch: {self._feature_arch}")

            # MLP for classification
            self._fc = FCLayers(self._d_dec, args.num_classes+1) # +1 for background

            # Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            # Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) # Imagenet mean and std
            ])

        def forward(self, x):
            x = self.normalize(x)
            batch_size, clip_len, channels, height, width = x.shape
            if self.training:
                x = self.augment(x)
            x = self.standarize(x)
            x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            # print(f"Permuted shape (B, C, T, H, W): {x.shape}")
            x = self._encoder(x)
            # print(f"Encoder output shape: {x.shape}") # [B, C, T', H, W]
            x = self._decoder(x)
            # print(f"Decoder output shape: {x.shape}") # [B, C, T, H, W]
            x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))
            # print(f"Pooled shape (temporal, 1, 1): {x.shape}")
            x = x.squeeze(-1).squeeze(-1)
            # print(f"Squeezed shape: {x.shape}")
            x = x.permute(0, 2, 1)  # [B, T, C]
            # print(f"Permuted shape (B, T, C): {x.shape}")
            im_feat = self._fc(x)
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
        # self._model.print_stats()
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
