"""
File containing the main model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from calflops import calculate_flops
from contextlib import nullcontext
from tqdm import tqdm
#Local imports
from model.modules import BaseRGBModel, FCLayers, step

from fvcore.nn import FlopCountAnalysis
from torchprofile import profile_macs

class GateShiftModule(nn.Module):
    """ Gate-Shift Fusion (GSF) to capture temporal relationships without significant overhead. """

    def __init__(self, channels, shift_ratio=0.25):
        super().__init__()
        self.channels = channels
        self.shift_channels = int(channels * shift_ratio)

        # Gate to control the importance of the shifted information
        self.gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: Input tensor (B, T, D)
        """
        B, T, D = x.shape
        x_shifted = x.clone()

        if T > 1:
            # Temporal shift: part of the channels are shifted forward and others backward
            x_shifted[:, 1:, :self.shift_channels] = x[:, :-1, :self.shift_channels]  # Shift forward
            x_shifted[:, :-1, self.shift_channels:2*self.shift_channels] = x[:, 1:, self.shift_channels:2*self.shift_channels]  # Shift backward
        
        # Apply the gate to fuse the shifted information
        gate_weights = self.gate(x)
        x = gate_weights * x + (1 - gate_weights) * x_shifted

        return x

class ShiftAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(ShiftAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.shift = 1  # Sequence shift

    def forward(self, x):
        # Apply the shift
        batch_size, seq_len, _ = x.shape
        shifted_x = torch.roll(x, shifts=self.shift, dims=1)  # Shift along the sequence

        # Apply attention on the shifted sequences
        attn_output, _ = self.attn(shifted_x, shifted_x, shifted_x)
        
        return attn_output
    
class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch

            # Switch RegNetY to EfficientNet
            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features

                # Remove final classification layer
                features.head.fc = nn.Identity()
                self._d = feat_dim
                
            elif self._feature_arch.startswith('effnet'):
                features = timm.create_model({
                    'effnet_b0': 'efficientnet_b0',
                    'effnet_b1': 'efficientnet_b1',
                    'effnet_b2': 'efficientnet_b2',
                    'effnet_b3': 'efficientnet_b3',
                    'effnet_b4': 'efficientnet_b4',
                    'effnet_b5': 'efficientnet_b5',
                    'effnet_b6': 'efficientnet_b6',
                    'effnet_b7': 'efficientnet_b7',
                }[self._feature_arch], pretrained=True)

                feat_dim = features.classifier.in_features  # Get output dimension

                # Remove final classification layer
                features.classifier = nn.Identity()
                self._d = feat_dim
            
            else:
                raise NotImplementedError(f"Model {args.feature_arch} not supported")

            self._features = features
            
            # Add GSF after feature extraction
            self.gsf = GateShiftModule(self._d)
            #self._shift_attention = ShiftAttention(d_model=self._d, nhead=8) 
            self._temp_rep = args.temp_rep
            if self._temp_rep == "lstm":
                # LSTM for temporal capture
                self._lstm = nn.LSTM(input_size=self._d, hidden_size=512, num_layers=2, batch_first=True)
                mlp_inp_size = 512
            elif self._temp_rep == "max_pool" or self._temp_rep == "avg_pool":
                mlp_inp_size = self._d
            elif self._temp_rep == "transformer":
                # Transformer for temporal capture
                #self._features_fc = nn.Linear(self._d, 512) 
                self._transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=self._d, nhead=8, dim_feedforward=2048),
                    num_layers=6
                )
                mlp_inp_size = self._d
            else:
                raise ValueError(f"Invalid value for temp_rep: {self._temp_rep}. Must be 'lstm', 'max_pool', 'avg_pool', or 'transformer'.")

            # MLP for classification
            self._fc = FCLayers(mlp_inp_size, args.num_classes)

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
            self.standarization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # ImageNet mean/std
            ])
        #     self.calculate_flopss(args)

        # def calculate_flopss(self, args):
        #     # Calculate FLOPs, MACs, and Params using calflops
        #     batch_size = 1
        #     input_shape = (batch_size, 3, 224, 398)  # Adjust this based on your input shape
        #     flops, macs, params = calculate_flops(model=self, 
        #                                           input_shape=input_shape,
        #                                           output_as_string=True,
        #                                           output_precision=4)
        #     print(f"FLOPs: {flops}   MACs: {macs}   Params: {params}")

        def forward(self, x):
            x = self.normalize(x)  # Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape  # B, T, C, H, W

            if self.training:
                x = self.augment(x)  # Batch augmentation

            x = self.standarize(x)  # ImageNet stats normalization
                        
            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d)  # B, T, D

            im_feat = self.gsf(im_feat)
            #im_feat = self._shift_attention(im_feat)
            if self._temp_rep == "lstm":
                # LSTM to capture temporal dependencies
                lstm_out, _ = self._lstm(im_feat)  # B, T, 512
                im_feat = lstm_out[:, -1, :]  # Take output of last time (last frame)
            elif self._temp_rep == "max_pool":
                # Max pooling over time
                im_feat = torch.max(im_feat, dim=1)[0]  # B, D
            elif self._temp_rep == "avg_pool":
                im_feat = torch.mean(im_feat, dim=1)  # B, D
            elif self._temp_rep == "transformer":
                #im_feat = self._features_fc(im_feat)
                # Transform input for the Transformer
                # Transformer expects a sequence of size (T, B, D), so we need to transpose
                im_feat = im_feat.permute(1, 0, 2)  # T, B, D
                # Pass through the Transformer
                transformer_out = self._transformer(im_feat)  # T, B, D
                # Take output of last frame
                im_feat = transformer_out[-1, :, :]  # B, D

            # Final MLP
            im_feat = self._fc(im_feat)  # B, num_classes

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
            print('Model params:', sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes
        
        self.count_parameters()

        # Memory estimation for inference
        self.estimate_inference_memory(self._model, torch.randn(args.batch_size, args.clip_len, 3, 224, 398).to(self.device), dtype='float32')

    def estimate_inference_memory(self, model, input_tensor, dtype='float32'):
        # Number of model parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Memory estimation for the model (in GB)
        param_size = 4 if dtype == 'float32' else 2  # float32: 4 bytes, float16: 2 bytes
        model_memory = (num_params * param_size) / 1e9  # In GB
        
        # Memory estimation for the input (in GB)
        input_size = input_tensor.numel() * param_size / 1e9  # In GB
        
        # Total memory estimation (only model and input)
        total_memory = model_memory + input_size  # In GB
        
        # print(f"Estimated model memory: {model_memory:.4f} GB")
        # print(f"Estimated input memory: {input_size:.4f} GB")
        # print(f"Estimated total memory for inference: {total_memory:.4f} GB")
        
        return total_memory

    def count_parameters(self):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            mem_gb = total_params * 4 / (1024 ** 3)
        
        # print(f"Total parameters: {total_params:,}")
        # print(f"Trainable parameters: {trainable_params:,}")
        # print(f"Estimated memory (float32): {mem_gb:.2f} GB")

    def parameters(self):
        """Redirects to the parameters of the internal model."""
        return self._model.parameters()


    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).float()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    loss = F.binary_cross_entropy_with_logits(pred, label)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)  # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:  # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # Apply sigmoid
            pred = torch.sigmoid(pred)
            
            return pred.cpu().numpy()

