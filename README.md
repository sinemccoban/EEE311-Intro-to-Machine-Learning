# EEE311-Intro-to-Machine-Learning
**Clothing Item Recognition with Machine Learning**

**Project Overview**

In this project, we aim to develop a machine learning model that can automatically identify and label clothing items worn by a person in an image. With the rapid growth of e-commerce and fashion-related applications, the ability to recognize and classify clothing items from images has become increasingly valuable.

**Motivation**

Understanding and categorizing clothing items in images has a wide range of applications, including:

Enhancing online shopping experiences by suggesting similar products to customers.
Automating the process of organizing and categorizing fashion catalogs.
Assisting visually impaired individuals in identifying their clothing.
Building personalized fashion recommendation systems.

**Features**

Our machine-learning model will offer the following features:

**Multi-Class Classification**: The model will be able to identify a wide variety of clothing items, such as shirts, pants, dresses, shoes, and accessories, and assign them the appropriate labels.

**High Accuracy**: We will strive to achieve high accuracy in classifying clothing items to ensure the model's practicality in real-world applications.

**Integration**: We will provide instructions for integrating the trained model into your own applications.

**Technologies Used**
**Python**: We will use Python as our primary programming language for model development.

**Google Colab**: The entire project will be developed in Google Colab, making it accessible and easy to run in the cloud.

**Deep Learning Frameworks**: We will leverage popular deep learning frameworks like TensorFlow or PyTorch to build, train, and evaluate our models.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Project Progress Report**


**1. Introduction**

In this project, the main goal is to develop a machine learning model for segmenting and classifying fashion images. The dataset comprises a large number of images and their corresponding fine-grained segmentations, with images named by a unique ImageId. The task involves identifying the types of clothing in the images, and the dataset includes both segmented apparel categories and fine-grained attributes. The challenge is to create a robust model that can accurately segment and classify clothing combinations in diverse poses.

**2. Related Work**

Prior research in image segmentation and classification has provided a foundation for this project. Notably, convolutional neural networks (CNNs) have proven successful in image-related tasks. Approaches like U-Net, SegNet, and ResNet have been widely employed for segmentation tasks. Additionally, research on fine-grained attribute recognition in fashion images has been considered for a comprehensive understanding of the problem.

**3. Employed Methodology**

Data Preprocessing: The training set, including images and corresponding annotations, is loaded and explored. The images are resized, normalized, and transformed. A custom dataset class is created to facilitate data loading and transformation.

Model Architecture: A CNN-based model is chosen for its effectiveness in image-related tasks. ResNet18, a pre-trained model, is used as the backbone. The last fully connected layer is replaced with a new layer for the specific classification task.

Training: The model is trained using the training dataset, and the cross-entropy loss function is employed. The Adam optimizer is utilized to update the model parameters. Training progress is monitored, and the model is evaluated on a validation set.

**4. Experimental Evaluation**

Dataset Split: The training dataset is split into training and validation sets to monitor the model's generalization performance.

Transformations: Image transformations, such as resizing and normalization, are applied to prepare the data for the model.

Model Training: The chosen CNN model is trained using the training set. The training loss is monitored to ensure convergence.

Validation: The model is evaluated on a separate validation set to assess its performance on unseen data. Validation accuracy is a key metric.

**5. Preliminary Results**

The model has been trained for five epochs, and preliminary results indicate promising performance. The training loss has steadily decreased, suggesting effective learning. In the validation phase, the accuracy has been monitored, providing insights into the model's ability to generalize to new, unseen data.

**6. Next Steps**

The following steps will be undertaken in the next phase of the project:

Fine-Tuning: The model will be fine-tuned to enhance its performance further.
Hyperparameter Tuning: Experimentation with different learning rates and batch sizes will be conducted to optimize the model's hyperparameters.
Data Augmentation: Augmenting the training dataset with additional transformations will enhance the model's ability to handle variations in pose and appearance.
Test Set Evaluation: The final model will be evaluated on the test set to assess its performance on completely unseen data.
In conclusion, the project is progressing well, with the initial training and validation phases showing promising results. The focus will now shift to refining the model and conducting a comprehensive evaluation on the test set. The ultimate aim is to develop a robust machine learning model capable of accurately segmenting and classifying fashion images, providing valuable insights into clothing combinations.

**The dataset we are using includes images related to clothing and fashion, along with associated segmentation information. The relevant files are as follows:**

train/ - Training Images: This folder contains images belonging to the training dataset.

test/ - Test Images: In this folder, you can find images from the test dataset. The segmentation and classification of these images will be performed.

train.csv - Training Labels: This file contains labels for the training dataset. It includes images with segmented clothing categories and fine-grained attributes, as well as images with only segmented clothing categories. It encompasses both scenarios.

label_descriptions.json - Description of Clothing Categories and Fine-grained Attributes: This file provides descriptions of clothing categories and fine-grained attributes.

sample_submission.csv - Sample Submission File in the Correct Format: An example submission file in the correct format.

The columns in the train.csv file include the following information:

ImageId: The identifier for an image.
EncodedPixels
ClassId: The class ID for this mask


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import os
import gc
import sys
import json
import glob
import random
from pathlib import Path

import cv2

csv_path = "/kaggle/input/imaterialist-fashion-2019-FGVC6/train.csv"
df = pd.read_csv(csv_path)

import matplotlib.pyplot as plt

import itertools
from tqdm import tqdm

from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold, KFold

with open("/kaggle/input/imaterialist-fashion-2019-FGVC6/label_descriptions.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]
df['CategoryId'] = df['ClassId'].str.split('_').str[0]
df['AttributeId'] = df['ClassId'].str.split('_').str[1:]

print("Total segments: ", len(df))


def show_img(img):
    I = cv2.imread("/kaggle/input/imaterialist-fashion-2019-FGVC6/train/" + img, cv2.IMREAD_COLOR)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    I = cv2.resize(I, (256, 256), interpolation=cv2.INTER_AREA)  
    plt.imshow(I)


def get_mask(df, img_id):
    a = df[df.ImageId == img_id]
    a = a.groupby('CategoryId', as_index=False).agg({'EncodedPixels':' '.join, 'Height':'first','Width':'first'})
    H = a.iloc[0, 2]
    W = a.iloc[0, 3]
    masks = []
    categories = []
    for line in a[['EncodedPixels', 'CategoryId']].iterrows():
        mask = np.full(H*W, dtype='int', fill_value=0)
        EncodedPixels = line[1][0]
        pixel_loc = list(map(int, EncodedPixels.split(' ')[0::2]))
        iter_num = list(map(int, EncodedPixels.split(' ')[1::2]))
        for p, i in zip(pixel_loc, iter_num):
            mask[p:(p+i)] = line[1][1]
        mask = mask.reshape(W, H).T
        masks += [mask]
        categories += [line[1][1]]
    return masks, categories


def new_mask(mask):
    matrix = [[0 for x in range(512)] for y in range(512)]
    for i in range(0, len(mask)):
        mask[i] = cv2.resize(mask[i], (512, 512), interpolation=cv2.INTER_NEAREST)
    for m in mask:
        for i in range(0, 512):
            for j in range(0, 512):
                if m[i][j] != 0:
                    matrix[i][j] = m[i][j]
    matrix = np.array(matrix)
    return matrix


def masked_image(df, image_id):
    masked_list, categories = get_mask(df, image_id)
    plt.figure(figsize=[30, 30])
    plt.subplot(1, 10, 1)
    I = cv2.imread("/kaggle/input/imaterialist-fashion-2019-FGVC6/train/" + image_id, cv2.IMREAD_COLOR)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    I = cv2.resize(I, (512, 512), interpolation=cv2.INTER_AREA)
    plt.imshow(I)
    plt.title('Input Image')
    i = 1
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    plt.imshow(I)
    for mask, cat in zip(masked_list, categories):
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        plt.subplot(1, 10, i+1)
        plt.imshow(mask)
        plt.title(label_names[int(cat)])
        plt.subplots_adjust(wspace=0.4, hspace=-0.65)
        i += 1
        if i == 10:
            break
    
    new_mask_ = new_mask(masked_list)
    I[new_mask_ == 0] = 0
    plt.imshow(I)


ids = df['ImageId'].unique()
for idx in ids[0:10]:
    masked_image(df, idx)


