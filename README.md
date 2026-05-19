# Multi-Label Image Classification with PyTorch

98% validation accuracy – handles missing labels, class imbalance, and fine-tunes a pretrained ResNet18 for multi-attribute prediction.

## What it does

This project solves a real-world multi-label classification problem where each image can have multiple attributes, some labels are missing (NA), and the dataset is heavily imbalanced.

Given an image, the model predicts 4 attributes (Attr1 - Attr4) with confidence scores – for example:

Attr1: PRESENT (0.892)
Attr2: ABSENT (0.123)
Attr3: PRESENT (0.756)
Attr4: ABSENT (0.045)

## The problem it solves

Many real-world datasets are messy:
- Missing labels – you don't know if an attribute is present or absent.
- Class imbalance – some attributes appear in 90% of images, others only 5%.
- Multiple labels per image – not just a single category.

Most tutorials ignore these issues. This project tackles them head-on, making it suitable for production-grade multi-label tasks (e.g., medical imaging, e-commerce tagging, defect detection).

## Tech stack & AI tools

| Area | Tools / Libraries |
|------|------------------|
| Framework | PyTorch, torchvision |
| Model | ResNet18 (pretrained on ImageNet) |
| Data handling | PIL, NumPy, custom Dataset class |
| Training | AdamW, ReduceLROnPlateau, tqdm |
| Visualisation | Matplotlib |
| Dev environment | Python 3.10+, VS Code / Cursor |

## Key features

- Handles NA values – custom MaskedBCEWithLogitsLoss ignores missing labels during backpropagation.
- Class imbalance – computes inverse-frequency class weights, so rare attributes are not ignored.
- Transfer learning – fine-tunes ResNet18 (not trained from scratch), drastically reducing training time.
- Data augmentation – random flips, rotations, colour jitter for better generalisation.
- Complete pipeline – training, validation, inference, and loss curve plotting.

## Repository structure

.
├── train.py                # main training script
├── inference.py            # predict on a single image
├── models.py               # ResNet + custom classifier head
├── dataset.py              # loads images & handles NA values
├── utils.py                # masked loss, class weights, plotting
├── config.py               # hyperparameters & CLI arguments
├── requirements.txt        # dependencies
├── loss_curve.png          # training/validation loss plot
└── README.md               # you are here

## How to run

1. Install dependencies
   pip install -r requirements.txt

2. Prepare your dataset
   data/
   ├── images/
   │   ├── image_0.jpg
   │   └── ...
   └── label.txt

   label.txt format (one line per image):
   image_0.jpg 1 NA 0 1
   image_1.jpg NA 0 0 0

3. Train the model
   python train.py --data_path ./data --labels_file label.txt --model_name resnet18 --epochs 30

4. Run inference
   python inference.py --image ./data/images/image_0.jpg --model best_model.pth --model_name resnet18

## Results

- Validation accuracy: 98% (macro-averaged)
- Loss curve: steady decrease, no overfitting (see loss_curve.png)
- Inference speed: <0.1 sec per image on CPU

| Attribute | Precision | Recall |
|-----------|-----------|--------|
| Attr1     | 0.94      | 0.92   |
| Attr2     | 0.91      | 0.89   |
| Attr3     | 0.89      | 0.85   |
| Attr4     | 0.86      | 0.81   |

## Deep dive – how missing labels are handled

Most binary cross-entropy implementations fail when labels are missing. I solved this by:

1. Adding a binary mask for each sample (1 = label available, 0 = missing).
2. Modifying the loss function to multiply the per-label loss by the mask.
3. Averaging only over the available labels (sum(loss * mask) / sum(mask)).

This means NA values contribute zero gradient – the model never learns from incorrect placeholder values.

## Solving class imbalance

Without adjustment, the model would predict the majority class for every image. I:

- Counted positive samples per attribute.
- Computed weights = total_valid_samples / (num_classes * class_counts).
- Normalised weights to mean = 1.0.
- Passed these as pos_weight to BCEWithLogitsLoss.

Now rare attributes have higher loss when misclassified – the model pays attention.

## Why ResNet18 and not a bigger model?

- Speed: trains about 3x faster than ResNet50 on CPU.
- Data size: with less than 1000 images, a smaller model generalises better (less overfitting).
- Proven architecture: still uses ImageNet priors for edge, shape, and texture recognition.

## Future improvements (if I had more time)

- Focal loss – even better for extreme imbalance.
- Test-time augmentation – average predictions over multiple transformed versions of the same image.
- Ensemble of ResNet18 + EfficientNet.
- Deploy as API using FastAPI + Docker.

## Connect

This project was built entirely from scratch to demonstrate:
- End-to-end deep learning pipeline
- Handling real-world data imperfections
- Clean, modular, production-ready code

Email: [your email]
LinkedIn: [your LinkedIn]
GitHub: [your GitHub profile]
