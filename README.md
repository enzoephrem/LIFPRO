**# Classification, Skull Stripping and Segmentation of Brain Tumor MRI

Projet : LIFPROJET 

Page de l'UE : ww. azdazid,pazd

## 1. Abstract


## 2. Introduction
Classification and segmentation of brain tumors are important tasks in the field of medical image analysis. These
techniques are used to accurately identify and locate tumors within brain images, which can help doctors diagnose and
treat patients with brain cancer. The classification of brain tumors involves assigning a specific type or category to a
tumor based on its characteristics, such as its size, shape, and appearance. Segmentation, on the other hand, involves
identifying the exact location and boundaries of a tumor within an image. Both tasks are typically performed using
specialized algorithms and software tools that are designed to analyze medical images and extract relevant information.
By accurately classifying and segmenting brain tumors, doctors can make more informed decisions about a patient’s
treatment and care. But to improve this segmentation and the doctors’ decisions, it is necessary to process as much data
as possible and remove what is not useful, such as the skull. The skull stripping allows first of all a better visual view of
the brain but also to reduce the errors of the segmentation. It is usually the radiologist’s job to do this, but it is a fastidious
and repetitive job that requires time and concentration. And like any repetitive work, it is possible to program it.

## 3. [Classification](Classification/)
### Dataset :
Kaggle, [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri), 

Files : [Classifier/dataset](Classification/dataset)

### Library Modules & Notebooks :

#### Modules
- [Classifier_Preprocessing.py](Classification/lib/Classifier_Preprocessing.py) :
This module contains the necessary functions for the preprocessing process of the classifier
  (notebook : [Preprocessing_classifier.ipynb](Prediction%2FPreprocessing_classifier.ipynb))

#### Notebook :
- [tumor_classifier.ipynb](Classification/tumor_classifier.ipynb) :
This notebook contains the training of the pre-trained research ([logs](Classification/logs))
- [tumor_classifier_Analyse.ipynb](Classification/tumor_classifier_Analyse.ipynb) :
This notebook analyse the logs of the research on the pre-trained models ([logs](Classification/logs))
- [tumor_classifier_advanced.ipynb](Classification/tumor_classifier_advanced.ipynb)
This notebook contains the training and the evaluation of the advanced classifier model. (saved model : [MobileNetV2_SGD_kl_divergence_50_2.h5](Classification/models-classification/)
- [Preprocessing_classifier.ipynb](Prediction%2FPreprocessing_classifier.ipynb) :
This notebook preprocess of the MRI to include in the advanced classifier model.
- [Classifier_prediction.ipynb](Prediction%2FClassifier_prediction.ipynb) :
Ths notbook contains the prediction test of the advanced classifer.

### Parameters of the advanced model:
- Loss function: kl divergence
- Optimizer: SGD
- Epochs: 50
- Batch size: 2
- Pre-trained model : VGG16


### Model Architecture of the advanced model:

#### VGG Architecture
<img src="Rapport/Ressources/vgg.png" width=75% height=75%>

#### Classifier Architecture
<img src="Rapport/Ressources/classifier.png" width=50% height=50%>

### Results of the advanced model :

During the project demo, I had this error : 

    2022-12-15 08:51:50.579432: E tensorflow/stream_executor/cuda/cuda_dnn.cc:389] Could not create cudnn handle: CUDNN_STATUS_NOT_INITIALIZED
    2022-12-15 08:51:50.579568: E tensorflow/stream_executor/cuda/cuda_dnn.cc:398] Possibly insufficient driver version: 515.86.1
    2022-12-15 08:51:50.579624: W tensorflow/core/framework/op_kernel.cc:1780] OP_REQUIRES failed at conv_ops_fused_impl.h:601 : UNIMPLEMENTED: DNN library is not found.

It's only a problem with my gpu card, I only had to relaunch jupyter-lab as I said, but for lack of time I did not do it.

- **Loss Progress**

![Loss Progress](Rapport/Ressources/loss_epochs.png "Loss Progress")

- **Accurency Progress**

![Accurency Progress](Rapport/Ressources/accurency_epochs.png "Accurency Progress")

- **Metrics**

|            |  training loss  |  training accuracy  | validation loss | validation accuracy |
|:----------:|:---------------:|:-------------------:|:---------------:|:-------------------:|
| **Values** |   0.0523        |    0.9842           |    0.1236       |       0.9624        |

- **Convolution Matrix**

![Convolution matrix](Rapport/Ressources/matrix_conv.png "Convolution matrix")



## 4. [Skull Stripping](Skull_Stripping/)

This model is a reuse and improvement of Muraligm Akshay’s model ([link](https://github.com/aksh-ai/skull-stripping-and-ica))
### Dataset :
Neurofeedback Skull-stripped, [NFBS repository](http://preprocessed-connectomes-project.org/NFB_skullstripped/)

Files : [Skull_Stripping/dataset_SkullStripping](Skull_Stripping/dataset_SkullStripping)

### Library Modules & Notebooks :
#### Notebook
- [Skull_Stripping_t1w.ipynb](Skull_Stripping/Skull_Stripping_t1w.ipynb) :
This notebook contains the training and the evaluation of the advanced skull stripping model. (saved model : [ResidualUNET3D_Adam_10_3.pth](Skull_Stripping/models_skull-stripping)
- [Skull-Stripping-T1_prediction.ipynb](Prediction/Skull-Stripping-T1_prediction.ipynb)
Ths notbook contains the prediction test of the advanced skull stripping model.

### Parameters of the advanced model:
- Loss function: Dice Loss
- Optimizer: Adam
- Epochs: 12
- Batch size: 2

### Model Architecture of the advanced model:

- **Residual Block** :

<img src="Skull_Stripping/ressources/residual_block_new.png" width=50% height=50%>

- **Upscale Block**

![upscale_block.png](Skull_Stripping%2Fressources%2Fupscale_block.png)

- **Residual Unet 3D advanced**

![upscale_block.png](Skull_Stripping/ressources/unet.png)


### Evaluation of the advanced model :

|            | MSE loss | Dice Score | IOU Score |
|:----------:|:--------:|:----------:|:---------:|
| **Values** |     0.019821     |  0.980179  | 0.961247  |


### Test Prediction of the advanced model :

![upscale_block.png](Rapport/Ressources/predict_skull_2.png)
![upscale_block.png](Rapport/Ressources/predict_skull_1.png)





















