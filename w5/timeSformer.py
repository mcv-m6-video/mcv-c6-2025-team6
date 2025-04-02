from torch.optim import AdamW
from video_transformers import VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead
from video_transformers.trainer import trainer_factory
import torch
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
backbone = TransformersBackbone("facebook/timesformer-base-finetuned-k400", num_unfrozen_stages=1)
base_path = r"C:/Users/laila/CVMasterActionRecognitionSpotting/SoccerNet/clips"
datamodule = VideoDataModule(
    train_root=f"{base_path}/train_root",
    val_root=f"{base_path}/val_root",
    batch_size=4,
    num_workers=0,
    num_timesteps=8,
    preprocess_input_size=224,
    preprocess_clip_duration=1,
    preprocess_means=backbone.mean,
    preprocess_stds=backbone.std,
    preprocess_min_short_side=256,
    preprocess_max_short_side=320,
    preprocess_horizontal_flip_p=0.5,
)

head = LinearHead(hidden_size=backbone.num_features, num_classes=12)
model = VideoModel(backbone, head)

optimizer = AdamW(model.parameters(), lr=1e-4)

Trainer = trainer_factory("single_label_classification")
trainer = Trainer(
    datamodule,
    model,
    optimizer=optimizer,
    max_epochs=8,  
)

def main():
    trainer.fit()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
