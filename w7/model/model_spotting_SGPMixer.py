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


class EDSGPMIXERLayers(nn.Module):
    '''Extracted from https://github.com/arturxe2/T-DEED/tree/main
    '''
    def __init__(self, feat_dim, clip_len, num_layers=1, ks=3, k=2, k_factor = 2, concat = True):
        super().__init__()
        self.num_layers = num_layers
        self.tot_layers = num_layers * 2 + 1
        self._sgp = nn.ModuleList(SGPBlock(feat_dim, kernel_size=ks, k=k, init_conv_vars=0.1) for _ in range(self.tot_layers))
        self._pooling = nn.ModuleList(nn.AdaptiveMaxPool1d(output_size = math.ceil(clip_len / (k_factor**(i+1)))) for i in range(num_layers))
        #self._upsample = nn.ModuleList(nn.Upsample(size = math.ceil(clip_len / (k_factor**i)), mode = 'linear', align_corners = True) for i in range(num_layers))
        self._sgpMixer = nn.ModuleList(SGPMixer(feat_dim, kernel_size=ks, k=k, init_conv_vars=0.1, 
                                        t_size = math.ceil(clip_len / (k_factor**i)), concat=concat) for i in range(num_layers))

    def forward(self, x):
        store_x = [] #Store the intermediate outputs
        #Downsample
        x = x.permute(0, 2, 1)
        for i in range(self.num_layers):
            x = self._sgp[i](x)
            store_x.append(x)
            x = self._pooling[i](x)
        
        #Intermediate
        x = self._sgp[self.num_layers](x)

        #Upsample
        for i in range(self.num_layers):
            x = self._sgpMixer[- (i + 1)](x = x, z = store_x[- (i + 1)])
            x = self._sgp[self.num_layers + i + 1](x)
        x = x.permute(0, 2, 1)

        return x
    
class SGPBlock(nn.Module):
    '''Extracted from https://github.com/arturxe2/T-DEED/tree/main
    '''
    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            init_conv_vars=0.1,  # init gaussian variance for the weight
            mode='normal'
    ):
        super().__init__()
        # must use odd sized kernel
        # assert (kernel_size % 2 == 1) and (kernel_size > 1)
        # padding = kernel_size // 2

        self.kernel_size = kernel_size

        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )


        self.act = act_layer()
        self.sigm = nn.Sigmoid()
        self.reset_params(init_conv_vars=init_conv_vars)

        self.mode = mode

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x):
        # X shape: B, C, T
        B, C, T = x.shape

        out = self.ln(x)
        psi = self.psi(out)
        fc = self.fc(out)
        convw = self.convw(out)
        convkw = self.convkw(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        if self.mode == 'normal':
            out = fc * phi + (convw + convkw) * psi + out #fc * phi instant level / (convw + convkw) * psi window level
        elif self.mode == 'sigm1':
            out = fc * phi + self.sigm(convw + convkw) * psi + out
        elif self.mode == 'sigm2':
            out = fc * self.sigm(phi) + self.sigm(convw + convkw) * psi + out
        elif self.mode == 'sigm3':
            out = self.sigm(fc) * phi + (convw + convkw) * self.sigm(psi) + out
        #out = fc * phi + out #only instant level
        #out = (convw + convkw) * psi + out #only window level
        #out = fc * phi + self.sigm(convw + convkw) * psi + out # sigmoid down branch window-level
        #out = fc * self.sigm(phi) + self.sigm(convw + convkw) * psi + out # sigmoid down branch window-level + up branch instant-level
        #out = self.sigm(fc) * phi + (convw + convkw) * self.sigm(psi) + out # sigmoid up branch window-level + down branch instant-level


        out = x + out
        # FFN
        out = out + self.mlp(self.gn(out))

        return out
    
class SGPMixer(nn.Module):
    '''Extracted from https://github.com/arturxe2/T-DEED/tree/main
    '''
    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            init_conv_vars=0.1,  # init gaussian variance for the weight
            t_size = 0,
            concat = True
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.concat = concat

        if n_out is None:
            n_out = n_embd

        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi1 = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.psi2 = nn.Conv1d(n_embd, n_embd, kernel_size = kernel_size, stride = 1, padding = kernel_size // 2, groups = n_embd)
        self.convw1 = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw1 = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        self.convw2 = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw2 = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)

        self.fc1 = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.global_fc1 = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        self.fc2 = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.global_fc2 = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        
        self.upsample = nn.Upsample(size = t_size, mode = 'linear', align_corners = True)

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        if self.concat:
            self.concat_fc = nn.Conv1d(n_embd * 6, n_embd, 1, groups = group)

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.psi2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc2.weight, 0, init_conv_vars)

        torch.nn.init.constant_(self.psi1.bias, 0)
        torch.nn.init.constant_(self.psi2.bias, 0)
        torch.nn.init.constant_(self.convw1.bias, 0)
        torch.nn.init.constant_(self.convkw1.bias, 0)
        torch.nn.init.constant_(self.convw2.bias, 0)
        torch.nn.init.constant_(self.convkw2.bias, 0)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.constant_(self.fc2.bias, 0)
        torch.nn.init.constant_(self.global_fc1.bias, 0)
        torch.nn.init.constant_(self.global_fc2.bias, 0)

        if self.concat:
            torch.nn.init.normal_(self.concat_fc.weight, 0, init_conv_vars)
            torch.nn.init.constant_(self.concat_fc.bias, 0)

    def forward(self, x, z):
        # X shape: B, C, T
        B, C, T = x.shape
        z = self.ln1(z)
        x = self.ln2(x)
        x = self.upsample(x)
        #x = self.ln2(x) # modified to have upsample inside sgp-mixer module (which seems more elegant)
        psi1 = self.psi1(z)
        psi2 = self.psi2(x)
        convw1 = self.convw1(z)
        convkw1 = self.convkw1(z)
        convw2 = self.convw2(x)
        convkw2 = self.convkw2(x)
        #Instant level branches
        fc1 = self.fc1(z)
        fc2 = self.fc2(x)
        phi1 = torch.relu(self.global_fc1(z.mean(dim=-1, keepdim=True)))
        phi2 = torch.relu(self.global_fc2(x.mean(dim=-1, keepdim=True)))

        out1 = (convw1 + convkw1) * psi1
        out2 = (convw2 + convkw2) * psi2
        out3 = fc1 * phi1
        out4 = fc2 * phi2

        if self.concat:
            out = torch.cat((out1, out2, out3, out4, z, x), dim = 1)
            out = self.act(self.concat_fc(out))

        else:
            out = out1 + out2 + out3 + out4 + z + x

        #out = z + out
        # FFN
        out = out + self.mlp(self.gn(out))

        return out

class LayerNorm(nn.Module):
    '''Extracted from https://github.com/arturxe2/T-DEED/tree/main
    '''
    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt)**self.gamma * CE_loss
        if self.alpha is not None:
            F_loss = self.alpha * F_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch
            # self._criterion = FocalLoss(gamma=2, alpha=None, reduction='mean') # Instancia Focal Loss

            if self._feature_arch == 'x3d_s' or self._feature_arch == 'x3d_m':
                x3d_full = torch.hub.load('facebookresearch/pytorchvideo', self._feature_arch, pretrained=True)
                x3d_full = freeze_temporal_stride(x3d_full) 
                features = nn.Sequential(*x3d_full.blocks[:5])
                self._d = 192  # output block 5
            else:
                raise NotImplementedError(f"Unsupported feature arch: {self._feature_arch}")

            self._features = features

            self._sgp_mixer = EDSGPMIXERLayers(
                            feat_dim=self._d,
                            clip_len=args.clip_len,
                            num_layers=2,
                            ks=9,
                            k=4,
                            concat=True
                        )

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
            
            # SGP
            x = self._sgp_mixer(x)  # [B, T, C]

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
                    # loss = self._model._criterion(pred, label) # Use this line for Focal Loss

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
