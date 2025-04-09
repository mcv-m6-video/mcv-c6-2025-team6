import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from contextlib import nullcontext
from tqdm import tqdm
import math
import timm
from model.modules import BaseRGBModel, FCLayers, step
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def freeze_temporal_stride(module):
    """Freeze temporal stride in Conv3d layers to preserve temporal resolution"""
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            if isinstance(m.stride, tuple) and m.stride[0] > 1:
                m.stride = (1, m.stride[1], m.stride[2])
    return module

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(BaseRGBModel):
    class Impl(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.args = args
            self._feature_arch = args.feature_arch
            self._d = args.transformer['d_model']  
            
            # Initialize backbone
            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                model_name = {
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]]
                features = timm.create_model(model_name, pretrained=True)
                backbone_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
                
            elif self._feature_arch.startswith('effnet'):
                model_name = {
                    'effnet_b0': 'efficientnet_b0',
                    'effnet_b1': 'efficientnet_b1',
                }[self._feature_arch]
                features = timm.create_model(model_name, pretrained=True)
                backbone_dim = features.classifier.in_features
                features.classifier = nn.Identity()
                
            elif self._feature_arch in ['x3d_s', 'x3d_m']:
                model_name = 'x3d_s' if self._feature_arch == 'x3d_s' else 'x3d_m'
                x3d_full = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
                x3d_full = freeze_temporal_stride(x3d_full)
                features = nn.Sequential(*x3d_full.blocks[:5])
                backbone_dim = 192

            else:
                features = models.video.__dict__['r3d_18'](pretrained=True)
                backbone_dim = features.fc.in_features
                features.fc = nn.Identity()
            
            self._features = features

            self.feature_proj = nn.Linear(backbone_dim, self._d)
            
            # Positional Encoding
            self.positional_encoding = PositionalEncoding(
                d_model=self._d,
                max_len=args.clip_len
            )
            
            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self._d,
                nhead=args.transformer['nhead'],
                dim_feedforward=args.transformer['dim_feedforward'],
                dropout=args.transformer['dropout'],
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=args.transformer['num_layers']
            )
            
            # Classification Head
            self.classifier = nn.Linear(self._d, args.num_classes + 1)
            
            # Augmentations
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])
            
            # Normalization
            self.normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

        def forward(self, x):
            B, T, C, H, W = x.shape
            
            # 1. Normalize and preprocess
            x = x / 255.0
            if self.training:
                x = self._apply_augmentations(x)
            x = self._apply_normalization(x)
            
            # 2. Backbone processing
            if self._feature_arch in ['x3d_s', 'x3d_m']:
                # X3D processing path
                x = x.permute(0, 2, 1, 3, 4)
                features = self._features(x)
                features = F.adaptive_avg_pool3d(features, (T, 1, 1))
                features = features.squeeze(-1).squeeze(-1).permute(0, 2, 1)
            else:
                # Other backbones
                if self._feature_arch.startswith(('rny', 'effnet')):
                    # 2D CNN processing
                    x = x.reshape(B*T, C, H, W)
                    features = self._features(x).reshape(B, T, -1)
                else:
                    # 3D CNN processing
                    x = x.permute(0, 2, 1, 3, 4)
                    if H != 112 or W != 112:
                        x = F.interpolate(x, size=(T, 112, 112), mode='trilinear')
                    features = self._features(x)
                    
                    # Handle different feature dimensions
                    if features.dim() == 5:
                        features = F.adaptive_avg_pool3d(features, (T, 1, 1))
                        features = features.squeeze(-1).squeeze(-1).permute(0, 2, 1)
                    elif features.dim() == 2:
                        features = features.unsqueeze(1).repeat(1, T, 1)
            
            # 3. Project features to consistent dimension
            features = features.reshape(B*T, -1)  # Flatten to [B*T, backbone_dim]
            features = self.feature_proj(features)  # Project to [B*T, transformer_dim]
            features = features.reshape(B, T, -1)
            
            # 4. Temporal modeling
            features = self.positional_encoding(features)
            temporal_features = self.transformer(features)
            
            # 5. Frame-level predictions
            logits = self.classifier(temporal_features)
            
            return logits

        def _apply_augmentations(self, x):
            B = x.shape[0]
            for i in range(B):
                x[i] = self.augmentation(x[i])
            return x

        def _apply_normalization(self, x):
            B = x.shape[0]
            for i in range(B):
                x[i] = self.normalize(x[i])
            return x

        def print_stats(self):
            total_params = sum(p.numel() for p in self.parameters())
            print(f"Model Parameters: {total_params:,}")
            print(f"Backbone: {self._feature_arch}")
            print(f"Transformer Layers: {self.args.transformer['num_layers']}")
            print(f"Attention Heads: {self.args.transformer['nhead']}")
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
    
            dummy_input = torch.randn(4, 50, 3, 224, 398).to(next(self.parameters()).device) 
    
            self.eval()
            try:
                flops = FlopCountAnalysis(self, dummy_input)
                print(f"FLOPs: {flops.total():,}")
                print(parameter_count_table(self))
            except Exception as e:
                print(f"Could not calculate FLOPs: {e}")

    def __init__(self, args=None):
        super().__init__()
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = self.Impl(args=args)
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
        
        with torch.no_grad() if inference else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frames = batch['frame'].to(self.device).float()
                labels = batch['label'].to(self.device).long()
                
                with torch.cuda.amp.autocast():
                    preds = self._model(frames)
                    loss = F.cross_entropy(
                        preds.reshape(-1, self._num_classes + 1),
                        labels.reshape(-1),
                        reduction='mean',
                        weight=weights
                    )
                
                if not inference:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)
                
                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)
            pred = torch.softmax(pred, dim=-1)
            return pred.cpu().numpy()

    def get_optimizer(self, optimizer_config):
        if self._args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=self._args.weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                self._model.parameters(),
                lr=optimizer_config['lr'],
                momentum=0.9,
                weight_decay=self._args.weight_decay
            )
        
        scaler = torch.cuda.amp.GradScaler(enabled=self._args.use_amp)
        return optimizer, scaler