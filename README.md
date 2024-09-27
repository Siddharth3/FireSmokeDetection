# FireSmokeDetection - using deep learning

### 1. Overview
#### Purpose:
Forest fires have become an increasing environmental threat, causing significant ecological and economic damage globally. Early detection is crucial in preventing widespread devastation and reducing response 
times. Traditional fire detection methods, such as temperature and smoke sensors, often suffer from limitations like delayed response, high costs, and poor coverage in large forest areas. To overcome these limitations, computer vision techniques combined with deep learning have emerged as powerful tools for fire and smoke detection. This project explores the use of deep learning algorithms for detecting forest fires using surveillance cameras, which offer a cost-effective and efficient solution.

#### Approach:
The project utilized deep learning models such as MobileNetV2, ResNet152, InceptionV3, InceptionResNetV2, and DenseNet169 to detect fire and smoke from forest images. The dataset used for training consisted of labeled images of smoke, fire, and non-fire scenarios, which were preprocessed and augmented to improve model robustness. The models were trained with performance metrics like categorical accuracy, recall, and precision to ensure high accuracy in detecting fire-related incidents. 

The workflow of the approach was as follows: 
###### Data Preprocessing 
Collected images were rescaled and augmented to improve generalization. Techniques such as horizontal flipping, rotation, and zoom were applied to simulate diverse real-world conditions. 
###### Model Selection and Training 
Multiple pre-trained deep learning models were fine-tuned using the forest fire dataset. These models were trained using the Adam optimizer and a categorical cross-entropy loss function. 
###### Verification and Evaluation 
The models were evaluated based on their validation accuracy, loss, precision, and recall. The best performing model (DenseNet169 in this case) was selected for further deployment. 
###### Prediction and Testing 
The selected model was tested on unseen data to predict fire and smoke occurrences. Performance was measured in terms of accuracy and the time taken for detection per image, demonstrating the model's practical application for real-time surveillance. 

This approach shows promising results and serves as a scalable framework for integrating deep learning with forest monitoring systems for early fire detection.

### 2. System Requirements
#### Hardware Requirements 
Processor: A multi-core processor (4 cores or more) is recommended for parallel computations, particularly if handling large datasets and training deep learning models. (Recommended: Intel Core i5/i7 or AMD Ryzen 5/7 series)

Memory (RAM): Minimum of 8 GB RAM is required. However, for training deep learning models and handling large datasets, 16 GB or more is preferred. 

GPU (Graphics Processing Unit): A dedicated GPU is highly recommended for deep learning tasks, as it speeds up the training of neural networks. (Recommended: NVIDIA GPUs with CUDA support, such as the NVIDIA GTX 1080 or RTX 2080 or newer, with at least 6-8 GB VRAM)

Storage: At least 50 GB of free storage for datasets, libraries, and models. SSD is preferred over HDD for faster data access. 

Display: A screen resolution of 1920x1080 or higher to visualize plots and performance metrics effectively. 
#### Software Requirements 
Operating System:

  - Recommended: Linux (Ubuntu 20.04 LTS or later), Windows 10, or macOS 10.14+. 

  - Preferred: Linux, for better compatibility with deep learning frameworks. 

Python Version: Python 3.6 or above (Python 3.8 or higher is recommended). 

Key Python Libraries:
  
  - TensorFlow: Version 2.x for deep learning model building and training. 
  
  - Keras: Integrated with TensorFlow for defining and training neural network models. 
  
  - Pandas: For data manipulation and analysis. 
  
  - Numpy: For numerical computations. 
  
  - Matplotlib and Seaborn: For data visualization. 
  
  - Scikit-learn: For preprocessing, splitting data, and performance metrics. 
  
  - Scikit-image: For reading and processing images. 

CUDA and cuDNN (if using an NVIDIA GPU): 
  
  - CUDA 11.x and cuDNN 8.x for GPU acceleration support with TensorFlow. 
#### Additional Libraries

Warnings and System Libraries: Libraries like warnings, os, sys, and time are essential for managing system interactions and handling warnings. 

TFSMLayer (TensorFlow Social Media Layer): Special TensorFlow layer for advanced deep learning tasks. 
#### Development Environment

Jupyter Notebook: Ideal for running Python code interactively, debugging, and visualizing results. 

Anaconda/Miniconda: For managing Python environments and dependencies easily. 
#### Internet Connection 
A stable internet connection is recommended for downloading datasets, pre-trained models, and dependencies from online repositories.

By meeting these hardware and software requirements, the system will be equipped to handle the project's tasks efficiently, especially when working with large datasets and training deep learning models.

### 4. Instructions for Use
If you installed above mentioned libraries and requirements are up to the mark, then you can directly run this script to test the sample and can see the output
```
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

classToLabel = {0: 'Smoke', 1: 'fire', 2: 'non fire'}

model = '/kaggle/input/densenet169/keras/default/1/densenet169_model.keras'
sampleImgPath = '/kaggle/input/forest-fire-and-non-fire-dataset/test/fire/Fire (1006).jpg'
# sampleImgPath = '/kaggle/input/forest-fire-and-non-fire-dataset/test/non fire/09363.jpg'
# sampleImgPath = '/kaggle/input/forest-fire-and-non-fire-dataset/test/Smoke/Smoke (1).jpg'

img_width = 299
img_height = 299

# predicting images
def predictOutput(model, path=True):
    img = image.load_img(sampleImgPath, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if path and sampleImgPath:
        model = k.models.load_model(model)
        images = np.vstack([x])
        eval_results = np.argmax(model.predict(x),axis=1)
        print(classToLabel[eval_results[0]])

predictOutput(model, sampleImgPath)
```

###### Indivdual may need to change the path as per their local/online system.

### 5. Contact Information
For any questions or issues related to the dashboard, please contact:

#### Dashboard Owner/Developer: Siddharth Sahni
#### Email: sidd.sahni3@gmail.com
#### LinkedIn: [Siddharth Sahni](https://www.linkedin.com/in/er-siddharth-sahni-36b227103/)
#### Website: [TheDataMan.github.io](https://siddharth3.github.io/TheDataMan.github.io/index.html)
