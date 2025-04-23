import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from contextlib import nullcontext
import torchvision.transforms as T
from model.modules import BaseRGBModel, step


class TemporalPyramidNetwork(nn.Module):
    def __init__(self, input_dim=192, num_classes=10, levels=3):
        super().__init__()
        self.levels = levels
        
        # Pyramid branches
        self.pyramid_branches = nn.ModuleList()
        for i in range(levels):
            stride = 2 ** i
            self.pyramid_branches.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, input_dim, kernel_size=3, 
                              stride=stride, padding=1),
                    nn.BatchNorm1d(input_dim),
                    nn.ReLU()
                )
            )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(input_dim * levels, input_dim, kernel_size=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Linear(input_dim, num_classes + 1)

    def forward(self, x):
        if x.dim() == 5:  # [B,C,T,H,W]
            x = x.mean(dim=[3,4])  # Global spatial average pooling
        
        if x.dim() == 4:  # [B,C,T,1]
            x = x.squeeze(-1)
        
        x = x.permute(0, 2, 1)  # [B,C,T] -> [B,T,C]
        
        # Multi-scale processing
        pyramid_features = []
        for branch in self.pyramid_branches:
            feat = branch(x.permute(0, 2, 1))  # [B,C,T]
            feat = F.interpolate(feat, size=x.shape[1], mode='nearest')
            pyramid_features.append(feat)
        
        # Fusion and classification
        fused = torch.cat(pyramid_features, dim=1)
        fused = self.fusion(fused).permute(0, 2, 1)  # [B,T,C]
        return self.classifier(fused)

class Model(BaseRGBModel):
    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch
            
            if self._feature_arch == 'x3d_s':
                x3d_full = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
                self._features = nn.Sequential(*x3d_full.blocks[:5])
                self._d = 192  # X3D-S feature dimension
            else:
                raise NotImplementedError(f"Unsupported feature arch: {self._feature_arch}")

            self._tpn = TemporalPyramidNetwork(
                input_dim=self._d,
                num_classes=args.num_classes,
                levels=3
            )
            
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
            self.standarization = T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )

        def forward(self, x):
            # Input: [B,T,C,H,W]
            x = self.normalize(x)
            if self.training:
                x = self.augment(x)
            x = self.standarize(x)
            
            # Process through X3D
            x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W]
            x = self._features(x)  # [B,192,T,7,7]
            
            # Process through TPN
            return self._tpn(x)  # [B,T,num_classes+1]

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

    def __init__(self, args=None):
        self.device = "cuda" if torch.cuda.is_available() and getattr(args, "device", "cpu") == "cuda" else "cpu"
        self._model = Model.Impl(args=args)
        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        is_training = optimizer is not None
        self._model.train() if is_training else self._model.eval()
        
        weights = torch.tensor([1.0] + [5.0] * self._num_classes, 
                             device=self.device)
        epoch_loss = 0.

        with torch.no_grad() if not is_training else nullcontext():
            for batch in tqdm(loader):
                frames = batch['frame'].to(self.device).float()
                labels = batch['label'].to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred = self._model(frames)
                    loss = F.cross_entropy(
                        pred.view(-1, self._num_classes + 1),
                        labels.view(-1),
                        weight=weights
                    )

                if is_training:
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    optimizer.zero_grad()
                    if lr_scheduler:
                        lr_scheduler.step()

                epoch_loss += loss.item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        self._model.eval()
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:  # (T,C,H,W)
            seq = seq.unsqueeze(0)
        seq = seq.to(self.device).float()

        with torch.no_grad(), torch.cuda.amp.autocast():
            pred = self._model(seq)
            return torch.softmax(pred, dim=-1).cpu().numpy()