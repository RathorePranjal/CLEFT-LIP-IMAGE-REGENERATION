# Cleft Lip Regeneration Toolkit: A Comprehensive College Project Report

## Abstract

This report presents a complete end-to-end implementation of a cleft lip image regeneration system using deep learning techniques. The project utilizes a mask-aware U-Net architecture trained on the CelebA dataset to perform inpainting tasks specifically targeted at reconstructing facial features affected by cleft lip conditions. The system includes training scripts, inference capabilities, and a web-based demo for practical application. This document covers all aspects of the project from conceptualization to deployment, including use cases, efficiency metrics, requirements, implementation details, workflows, and file explanations.

## 1. Project Understanding

### 1.1 Overview
The Cleft Lip Regeneration Toolkit is an AI-powered system designed to reconstruct facial images by regenerating areas affected by cleft lip deformities. It employs computer vision and deep learning techniques to perform image inpainting, where missing or damaged regions of an image are filled in realistically based on surrounding context.

### 1.2 Core Technology
- **Architecture**: Mask-aware U-Net neural network
- **Training Data**: CelebA dataset (aligned facial images)
- **Loss Functions**: Combination of L1 loss, perceptual loss, and mask-weighted loss
- **Framework**: PyTorch with CUDA acceleration
- **Deployment**: FastAPI web application with HTML/CSS/JavaScript frontend

### 1.3 Key Features
- Offline operation (no external API dependencies)
- Real-time web demo with interactive mask drawing
- Configurable training parameters
- Evaluation metrics (PSNR, SSIM)
- Checkpoint saving and resuming capabilities

## 2. Use Case

### 2.1 Primary Applications
1. **Medical Imaging**: Assist healthcare professionals in visualizing post-surgical outcomes for cleft lip patients
2. **Research and Education**: Provide tools for studying facial reconstruction techniques
3. **Pre-surgical Planning**: Help surgeons plan reconstructive procedures
4. **Patient Education**: Demonstrate potential treatment outcomes to patients and families

### 2.2 Target Users
- Plastic surgeons and maxillofacial specialists
- Medical researchers
- Students studying computer vision and medical imaging
- Healthcare institutions implementing AI-assisted diagnostics

### 2.3 Real-world Impact
The system addresses the challenge of visualizing treatment outcomes for cleft lip patients, potentially improving patient counseling, surgical planning, and overall treatment efficacy. By providing realistic reconstructions, it helps bridge the gap between current condition and expected results.

## 3. Efficiency

### 3.1 Performance Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality (higher is better)
- **SSIM (Structural Similarity Index)**: Evaluates structural preservation (range: 0-1, higher is better)
- **Training Time**: ~2-4 hours for 50 epochs on GPU (depending on hardware)
- **Inference Speed**: <1 second per image on GPU

### 3.2 Hardware Utilization
- **GPU Memory**: ~2-4 GB during training (batch size dependent)
- **CPU Usage**: Minimal during inference, moderate during training
- **Storage**: ~1GB for trained model, ~10GB for dataset

### 3.3 Optimization Features
- Mixed precision training (FP16) for faster convergence
- Data augmentation (horizontal flipping)
- Gradient scaling for numerical stability
- Automatic checkpoint resuming

### 3.4 Scalability
- Batch processing capabilities
- Configurable image resolutions (256x256 to 512x512)
- Modular architecture for easy extension

## 4. Requirements

### 4.1 Hardware Requirements
- **Minimum**: CPU with 8GB RAM, 20GB storage
- **Recommended**: NVIDIA GPU with 4GB+ VRAM, 16GB RAM, 50GB storage
- **Optimal**: NVIDIA GPU with 8GB+ VRAM, 32GB RAM, SSD storage

### 4.2 Software Requirements
- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher
- **Dependencies**:
  - PyTorch >= 2.4.0
  - Torchvision >= 0.19.0
  - NumPy >= 1.26.0
  - Pillow >= 10.0.0
  - OpenCV-Python >= 4.10.0
  - Scikit-Image >= 0.24.0
  - Tqdm >= 4.66.0
  - FastAPI >= 0.110.0
  - Uvicorn >= 0.30.0
  - Python-Multipart >= 0.0.9
  - Jinja2 >= 3.1.0

### 4.3 Data Requirements
- **CelebA Dataset**: Aligned facial images (download from official source)
- **Mask Dataset**: Binary PNG masks (255 = regenerate, 0 = preserve)
- **Image Format**: RGB images, PNG/JPG formats supported

## 5. How to Implement and Run

### 5.1 Environment Setup
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Data Preparation
1. Download CelebA aligned images
2. Extract to `data/celeba/` directory
3. Prepare irregular masks in `data/masks/` directory
4. Ensure filename correspondence between images and masks

### 5.3 Training
```bash
python -m src.train \
  --image-root data/celeba \
  --mask-root data/masks \
  --epochs 50 \
  --batch-size 8 \
  --save-dir checkpoints \
  --use-perceptual \
  --lambda-mask 10.0
```

### 5.4 Inference
```bash
python -m src.infer \
  --weights artifacts/best.pt \
  --image path/to/input.jpg \
  --mask path/to/mask.png \
  --output output.png
```

### 5.5 Web Demo
```bash
# Copy best model to artifacts/
cp checkpoints/epoch_050.pt artifacts/best.pt

# Start server
uvicorn web.main:app --reload --port 8000

# Access at http://127.0.0.1:8000
```

## 6. Implementation Details

### 6.1 Architecture Overview
The system consists of four main components:
1. **Data Pipeline**: Custom dataset class for loading and preprocessing
2. **Model Architecture**: U-Net with mask-aware input
3. **Training Loop**: Multi-loss optimization with validation
4. **Inference Engine**: Single-image processing pipeline
5. **Web Interface**: FastAPI backend with HTML frontend

### 6.2 U-Net Architecture
- **Input**: 4 channels (RGB + binary mask)
- **Encoder**: 4 downsampling blocks (64 → 128 → 256 → 512 → 1024 channels)
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: 3 channels (RGB reconstruction)
- **Activation**: Sigmoid for pixel value normalization

### 6.3 Loss Functions
1. **Pixel Loss**: L1 distance between predicted and ground truth
2. **Perceptual Loss**: VGG16-based feature matching
3. **Mask-Weighted Loss**: Higher weight on masked regions

### 6.4 Training Strategy
- **Optimizer**: Adam with learning rate 1e-4
- **Scheduler**: None (constant LR)
- **Batch Size**: 8 (configurable)
- **Validation**: 5% of dataset
- **Checkpointing**: Best model saved based on validation loss

### 6.5 Data Augmentation
- Random horizontal flipping (50% probability)
- Resize and crop to target resolution
- Normalization to [0,1] range

## 7. Flowchart

### 7.1 System Architecture Flowchart
```
[Data Input]
    │
    ▼
[Preprocessing]
    │
    ▼
[Model Training/Inference]
    │
    ▼
[Post-processing]
    │
    ▼
[Output Generation]
```

### 7.2 Training Pipeline Flowchart
```
[Load Dataset] → [Data Augmentation] → [Model Forward Pass]
    │                    │                        │
    ▼                    ▼                        ▼
[Validation Split] → [Loss Calculation] → [Backpropagation]
    │                    │                        │
    ▼                    ▼                        ▼
[Metrics Logging] → [Checkpoint Saving] → [Next Epoch]
```

### 7.3 Inference Pipeline Flowchart
```
[Input Image + Mask] → [Preprocessing] → [Model Inference]
    │                        │                    │
    ▼                        ▼                    ▼
[Validation] → [Normalization] → [Post-processing]
    │                        │                    │
    ▼                        ▼                    ▼
[Output Image] ← [Denormalization] ← [Save Result]
```

## 8. Workflow

### 8.1 Development Workflow
1. **Data Collection**: Gather CelebA images and create corresponding masks
2. **Environment Setup**: Install dependencies and configure GPU
3. **Model Development**: Implement U-Net architecture and training loop
4. **Training**: Run training script with hyperparameter tuning
5. **Evaluation**: Assess model performance using PSNR/SSIM metrics
6. **Inference Testing**: Validate single-image reconstruction quality
7. **Web Deployment**: Integrate model into FastAPI application
8. **User Interface**: Develop HTML/CSS/JS frontend for interaction

### 8.2 User Workflow
1. **Access Web Demo**: Open browser to localhost:8000
2. **Upload Image**: Select facial image for reconstruction
3. **Create Mask**: Draw white regions to indicate areas for regeneration
4. **Submit Request**: Click regenerate button to process
5. **View Results**: Compare original and reconstructed images
6. **Download Output**: Save regenerated image for further use

### 8.3 Maintenance Workflow
1. **Model Updates**: Retrain with new data or improved architectures
2. **Performance Monitoring**: Track inference speed and quality metrics
3. **Bug Fixes**: Address issues in data pipeline or model inference
4. **Feature Additions**: Implement new capabilities (e.g., batch processing)

## 9. Explanation of All Files

### 9.1 Root Directory Files
- **README.md**: Comprehensive project documentation and usage guide
- **requirements.txt**: Python dependencies with version specifications
- **train_colab.ipynb**: Jupyter notebook for Google Colab training environment

### 9.2 Source Code (`src/` directory)
- **train.py**: Main training script with argument parsing and training loop
- **infer.py**: Command-line inference tool for single images
- **data/dataset.py**: PyTorch Dataset class for CelebA + mask loading
- **models/unet.py**: U-Net architecture implementation with loading utilities
- **utils/metrics.py**: PSNR and SSIM evaluation functions

### 9.3 Web Application (`web/` directory)
- **main.py**: FastAPI application with upload/inference endpoints
- **templates/index.html**: HTML interface with canvas-based mask drawing
- **static/styles.css**: CSS styling for responsive web design

### 9.4 Data and Artifacts
- **data/celeba/**: Directory for CelebA facial images
- **data/masks/**: Directory for binary mask images
- **checkpoints/**: Training checkpoint storage (.pt files)
- **artifacts/**: Final model weights and exported models

## 10. Flowcharts Used

### 10.1 U-Net Architecture Diagram
```
Input (4ch) → ConvBlock → Down → Down → Down → Down
    │              │         │      │      │      │
    │              │         │      │      │      │
    │              └─────────┼──────┼──────┼──────┘
    │                        │      │      │
    ▼                        ▼      ▼      ▼
Output (3ch) ← ConvBlock ← Up ← Up ← Up ← Up
```

### 10.2 Data Processing Pipeline
```
Raw Images → Resize → ToTensor → Normalize → Model Input
Mask Images → Resize → ToTensor → Binarize → Model Input
```

### 10.3 Training Loop Diagram
```
For each epoch:
  For each batch:
    Forward pass → Loss calculation → Backward pass → Optimizer step
  Validation → Metrics logging → Checkpoint saving
```

### 10.4 Web Request Flow
```
User Upload → File validation → Preprocessing → Model inference → Post-processing → JSON response
```

## 11. Inspirations and References

### 11.1 Academic Papers
- **U-Net: Convolutional Networks for Biomedical Image Segmentation** (Ronneberger et al., 2015)
  - Inspired the core architecture for medical image segmentation and reconstruction
- **Image Inpainting for Irregular Holes Using Partial Convolutions** (Liu et al., 2018)
  - Provided foundation for mask-aware inpainting techniques
- **Perceptual Losses for Real-Time Style Transfer and Super-Resolution** (Johnson et al., 2016)
  - Influenced the use of perceptual loss for improved image quality

### 11.2 Open-Source Projects
- **PyTorch Vision Models**: Utilized VGG16 for perceptual loss implementation
- **FastAPI Examples**: Guided the web application structure
- **CelebA Dataset**: Primary training data source from Liu et al. (2015)

### 11.3 Code Attribution
All code in this project is original implementation inspired by the above research. No direct code copying was performed. The U-Net architecture follows standard implementations commonly found in computer vision literature, adapted specifically for the cleft lip inpainting task.

## 12. Challenges and Solutions

### 12.1 Technical Challenges
1. **Mask Alignment**: Ensuring proper correspondence between images and masks
   - **Solution**: Filename-based pairing with cycling for unequal datasets
2. **Memory Constraints**: GPU memory limitations during training
   - **Solution**: Gradient accumulation and mixed precision training
3. **Artifact Generation**: Unnatural reconstructions around mask boundaries
   - **Solution**: Mask-weighted loss and perceptual regularization

### 12.2 Implementation Challenges
1. **Data Preparation**: Creating realistic cleft lip masks
   - **Solution**: Manual annotation and irregular mask generation
2. **Model Convergence**: Training stability issues
   - **Solution**: Careful hyperparameter tuning and loss weighting
3. **Web Deployment**: Real-time inference optimization
   - **Solution**: Model quantization and efficient preprocessing

## 13. Future Enhancements

### 13.1 Model Improvements
- Integration of attention mechanisms
- Multi-scale feature fusion
- Conditional generation with style control

### 13.2 Application Extensions
- Batch processing capabilities
- Video frame reconstruction
- Mobile application deployment

### 13.3 Research Directions
- Clinical validation studies
- Comparison with traditional reconstruction methods
- Extension to other facial deformities

## 14. Conclusion

The Cleft Lip Regeneration Toolkit represents a comprehensive implementation of deep learning for medical image reconstruction. By combining state-of-the-art computer vision techniques with practical deployment considerations, the project demonstrates the potential of AI in healthcare applications. The modular architecture, thorough documentation, and web-based interface make it accessible for both research and practical use.

Key achievements include:
- Successful implementation of mask-aware inpainting
- Real-time web demo with interactive features
- Comprehensive evaluation framework
- End-to-end workflow from training to deployment

This project serves as a foundation for further research in AI-assisted medical imaging and demonstrates the importance of interdisciplinary approaches combining computer science and healthcare expertise.

## 15. References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer.

2. Liu, G., Reda, F. A., Shih, K. J., Wang, T. C., Tao, A., & Catanzaro, B. (2018). Image inpainting for irregular holes using partial convolutions. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 85-100).

3. Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual losses for real-time style transfer and super-resolution. In European conference on computer vision (pp. 694-711). Springer.

4. Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision (pp. 3730-3738).

---

**Project Author**: [Your Name]  
**Institution**: [Your College/University]  
**Date**: [Current Date]  
**Version**: 1.0
