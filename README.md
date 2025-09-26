# Leaf Disease Classification - LiteLeafNet

This project provides a **lightweight and efficient pipeline for leaf disease detection**, splitting the task into two subtasks:

1. **Disease presence/absence classification**  
2. **Disease class classification**

It leverages lightweight models on the device side and powerful models on the server side to balance performance and computational efficiency.

---

## Features

- Binary classification (disease presence/absence) using **MobileNetV2** on the device  
- Multi-class classification (disease type) using **ResNet50** on the server  
- **Grad-CAM-based heatmap masking** to reduce irrelevant image regions before server transmission  
- Improvements:
    - **20.80% reduction** in computational cost
    - **13.91% enhancement** in the accuracy-cost trade-off compared to methods that transmit full images

---

## Dataset

- Dataset Preparation: 
- Go to: https://github.com/spMohanty/PlantVillage-Dataset

- Download the repository
- Go to: ./PlantVillage-Dataset/raw/color

- Change the dataset style as follows:
    - ./PlantVillage-Dataset/raw/color/disease/<disease folders...>
    - ./PlantVillage-Dataset/raw/color/healthy/<healthy folders...>
