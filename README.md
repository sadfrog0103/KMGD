🚀Unified Knowledge Distillation Framework

English Introduction
Unified KD Framework is a flexible, configuration-driven PyTorch codebase designed for Knowledge Distillation (KD) research. It supports standard distillation methods (like Hinton KD) as well as advanced Unified Prototype-based Distillation.

The framework is designed to be modular, extensible, and easy to reproduce, featuring modern training tricks like Mixup, AMP (Automatic Mixed Precision), and Cosine Annealing.

✨ Key Features
🧩 Modular Design: Decoupled training engine, models, data pipeline, and loss functions.

🔥 Distillation Methods:
Hinton KD: Standard Logits-based distillation.
Unified KD: A structural alignment method using learnable prototypes, orthogonal constraints, and feature projection.

🏗️ Model Zoo:
ResNet: ResNet-18/34/50/101/152 (CIFAR & ImageNet optimized).
MobileNetV2: Efficient edge-side models.
Vision Transformer (ViT): ViT-Tiny/Small/Base with adaptive patch embedding for small images.

⚡ SOTA Training Tricks:
Data Augmentation: Mixup, RandomCrop, HorizontalFlip, Normalization.
Optimization: Label Smoothing, Cosine Learning Rate Scheduler with Warmup, AMP (Mixed Precision).
📊 Visualization: Built-in support for TensorBoard logging and prototype heatmap visualization.
💻 Platform Friendly: optimized for both Linux and Windows (solved multiprocessing dataloader issues).


📂 Project Structure
kd_research/
├── configs/               # YAML Configuration files
│   ├── cifar100_r18_single.yaml
│   ├── cifar100_r18_hinton.yaml
│   └── cifar100_r18_unified.yaml
├── data/                  # Dataset handlers
│   └── datasets.py        # CIFAR10/100, STL10, ImageNet wrappers
├── engine/                # Training Logic
│   └── trainer.py         # One epoch training & validation
├── losses/                # Loss Functions
│   ├── builder.py         # Loss factory
│   └── unified_kd.py      # Unified KD implementation
├── models/                # Model Definitions
│   ├── base_model.py      # Abstract base class
│   ├── resnet.py          # ResNet family
│   ├── mobilenet.py       # MobileNetV2
│   └── vit.py             # Vision Transformer
├── tools/                 # Entry Points
│   └── train.py           # Main training script
└── utils/                 # Utilities
    ├── checkpoint.py      # Safe checkpoint loading
    ├── config.py          # YAML parser
    ├── logger.py          # Logging setup
    └── mixup.py           # Mixup implementation

🚀 Usage
All experiments are driven by YAML configuration files. You don't need to change code to switch models or methods.

1. Train a Teacher/Student Alone (Single Mode)
To train a baseline model (e.g., ViT-Tiny or ResNet-18) from scratch: 
python tools/train.py --config configs/cifar100_vit_tiny_single.yaml

2. Knowledge Distillation
Option A: Standard Hinton KD
python tools/train.py --config configs/cifar100_r18_hinton.yaml

Option B: Unified KD (Our Method)
python tools/train.py --config configs/cifar100_r18_unified.yaml

📊 Configuration Guide
The configuration file controls everything. Key parameters:
Parameter,Description,
mode,single or distill,
data.dataset,"cifar100, cifar10, imagenet",
model.student,"resnet18, vit_tiny, mobilenetv2...",
solver.amp,true / false,Enable Automatic Mixed Precision 
solver.mixup,0.8 (float),Mixup alpha value (Set 0 to disable)
distillation.method,"hinton, unified",Distillation algorithm

🤝 Contributing 
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 欢迎提交 PR。对于重大更改，请先提交 Issue 讨论。
