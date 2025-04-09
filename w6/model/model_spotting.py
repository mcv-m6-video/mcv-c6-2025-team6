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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe) 

    def forward(self, x):
        # x shape: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T]
    
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return x + self.position_embeddings[:, :seq_len, :]
    
class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch
            self._use_tcn= args.use_tcn
            self._use_pe=args.use_pe
            self._use_pe_learnable=args.use_pe_learnable
            self._use_attn=args.use_attn

            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
                self._d = feat_dim

            elif self._feature_arch.startswith('effnet'):
                features = timm.create_model({
                    'effnet_b0': 'efficientnet_b0',
                    'effnet_b1': 'efficientnet_b1',
                }[self._feature_arch], pretrained=True)

                feat_dim = features.classifier.in_features 
                features.classifier = nn.Identity()
                self._d = feat_dim
            elif self._feature_arch == 'x3d_s':
                x3d_full = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
                # Keep only the convolutional blocks (avoid classification head)
                #print(x3d_full)
                x3d_full=freeze_temporal_stride(x3d_full)
                features = nn.Sequential(*x3d_full.blocks[:5])  # Up to before the final projection
                #features = nn.Sequential(*x3d_full.blocks[:4])  # Up to before the block 5
                self._d = 192  # output block 5
                #self._d =96 # output block 4
            elif self._feature_arch == 'x3d_m':
                x3d_full = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
                #print(x3d_full)
                x3d_full=freeze_temporal_stride(x3d_full)
                features = nn.Sequential(*x3d_full.blocks[:5]) # Up to before the final projection
                #features = nn.Sequential(*x3d_full.blocks[:4])  # Up to before the block 5 
                self._d = 192 # output block 5
                #self._d =96 # output block 4
            else:
                raise NotImplementedError(f"Unsupported feature arch: {self._feature_arch}")

            self._features = features
            if self._use_pe:
                self._pe = PositionalEncoding(self._d)

            if self._use_pe_learnable:
                self._pe = LearnablePositionalEncoding(self._d)

            if self._use_tcn:
                self._tcn = nn.Sequential(
                    nn.Conv1d(self._d, self._d, kernel_size=3, padding=1),  # preserves T
                    nn.ReLU(),
                    nn.Conv1d(self._d, self._d, kernel_size=3, padding=1),
                    nn.ReLU()
                )
            if self._use_attn:
                self._attn = nn.MultiheadAttention(embed_dim=self._d, num_heads=4, batch_first=True)
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
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats
            
            if self._feature_arch=='x3d_s' or  self._feature_arch=='x3d_m':          
                x = x.permute(0, 2, 1, 3, 4)

                # Pass through X3D backbone (keeps temporal info)
                x = self._features(x) 

                x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))  # (B, D, T, 1, 1)
                x = x.squeeze(-1).squeeze(-1)  # (B, D, T)
                x = x.permute(0, 2, 1)  # (B, T, D)
            else:
                x = self._features(
                    x.view(-1, channels, height, width)
                ).reshape(batch_size, clip_len, self._d) #B, T, D

            if self._use_pe: # positional sinusoidal encoding
                x = self._pe(x)

            if self._use_pe_learnable: #positional learnable encoding
                x = self._pe(x)

            if self._use_attn: #multihead:attention
                x, _ = self._attn(x, x, x)

            if self._use_tcn: #temporal convolutional network
                x = x.permute(0, 2, 1)  # (B, D, T)
                x = self._tcn(x)
                x = x.permute(0, 2, 1)  # back to (B, T, D)
            #MLP
            im_feat = self._fc(x) #B, T, num_classes+1

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
