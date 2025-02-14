# **Segmentation-Based Bird Audio Denoising**

[Segmentation-Based Bird Audio Denoising](https://www.researchgate.net/publication/387187819_Segmentation-Based_Bird_Audio_Denoising)

## **Overview**
Bird sound denoising in natural environments is challenging due to background noise such as wind, water, and other environmental disturbances. This project extends a **segmentation-based approach** to **audio denoising**, following the framework proposed by Zhang et al. (2022), which treats **audio denoising as an image segmentation problem**.

By converting bird audio signals into **spectrogram images** using the **Short-Time Fourier Transform (STFT)**, a deep segmentation model isolates clean bird calls from noisy backgrounds. Our model achieves an **Intersection over Union (IoU) score of 63%**, demonstrating its effectiveness in real-world denoising tasks.

## **Key Features**
- **Segmentation-Based Audio Denoising**: Reformulates the denoising problem as an image segmentation task.
- **STFT Spectrogram Transformation**: Converts audio signals into images for deep learning-based processing.
- **Deep Learning-Based Segmentation**: Uses a **ResNet50-based encoder-decoder model** for segmenting spectrograms.
- **Improved Denoising Performance**: Achieves an **IoU of 63%**, comparable to leading architectures in the field.
- **Robust Against Natural Noise**: Handles complex background noise in outdoor bird recordings.

## **Dataset**
- **BirdSoundsDenoising Dataset** (Zhang et al. 2022)
- **1,500 spectrograms** (1,000 train / 200 validation / 300 test)
- Each image is labeled with **binary segmentation masks** indicating clean bird sounds vs. noise.
- Spectrograms resized to **224×224 pixels** for model input.

## **Methodology**
1. **Audio Preprocessing**:
   - Convert bird sound recordings into spectrograms using **Short-Time Fourier Transform (STFT)**.
   - Apply **contrast enhancement** to highlight noise patterns.
   - **Data augmentation** (random flipping) to improve generalization.

2. **Model Architecture**:
   - **Encoder**: **ResNet50** pre-trained on ImageNet for feature extraction.
   - **Decoder**: Four **transpose convolution layers** with skip connections.
   - **Final Layer**: **1×1 convolution** followed by a **sigmoid activation** for binary segmentation.

3. **Loss Function**:
   - **Binary Cross-Entropy (BCE) Loss**
   - **Dice Loss** for handling class imbalance.

4. **Training Configuration**:
   - **Optimizer**: Adam with a learning rate of **1×10⁻⁴**.
   - **Batch Size**: 16.
   - **Epochs**: 20 with early stopping.

## **Results**
| Model        | IoU (%) |
|-------------|--------|
| U²-Net      | 44.8   |
| MTU-Net     | 55.7   |
| Segmenter   | 57.7   |
| **Our Model**  | **63.0**  |
| U-Net       | 63.9   |
| SegNet      | 65.3   |
| DeepLabV3   | 72.3   |

- Our model achieves **63.0% IoU**, **outperforming several widely used architectures**.
- **Qualitative improvements**: Cleaner, sharper audio signals compared to baseline models.
- **Future Enhancements**: Incorporate **attention mechanisms** or **hybrid CNN-ViT architectures** to improve segmentation accuracy.

## **Installation & Usage**
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/bird-audio-denoising.git
cd bird-audio-denoising
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Jupyter Notebook**
Open the `.ipynb` file in **Jupyter Notebook** or **Google Colab** and execute the cells to preprocess data, train the model, and evaluate performance.

## **Future Work**
- **Optimize model architecture** with multi-scale feature extraction and self-attention layers.
- **Improve inference speed** for real-time audio denoising applications.
- **Expand dataset** to include more diverse bird species and environmental noise conditions.
