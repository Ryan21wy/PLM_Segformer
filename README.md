# PLM_Segformer: Automatic Damage Segmentation of Sanskrit Palm-leaf Manuscripts

PLM-Segformer framework is developed to provide an automated damage segmentation method for Sanskrit PLMs, which builds on the original Segformer architecture. The hyperparameters for pre-processing, training, inference, and post-processing phases are fully optimized to make the original model more suitable for the PLM segmentation task. The model has been used for automated PLM damage detection in Potala Palace, and it can complete 10064 pages of PLM damage Segmentation within 24 hours.

![WorkFlow](https://user-images.githubusercontent.com/81405754/223337166-b757a1e2-a28d-4d41-b66c-1c6b281efb9b.png)

Flowchart of the PLM damage segmentation method. (a) The PLM dataset is established by digital camera acquisition and manual annotation. It has been subsequently divided into the training set, validation set, and test set. Then, various pre-processing methods (b) and loss functions (c) are compared to find the best way to build the damage segmentation models. Finally, inference enhancement methods (d) and post-processing methods (e) were used to optimize the prediction results.

## Development version
  
1. Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)   
2. Install [Git](https://git-scm.com/downloads)  
4. Open commond line, create environment and enter with the following commands:  

        conda create -n PLM_Segformer python=3.8
        conda activate PLM_Segformer

5. Clone the repository and enter:  

        git clone https://github.com/Ryan21wy/PLM_Segformer.git
        cd PLM_Segformer

6. Install dependency with the following commands:  
        
        pip install -r requirements.txt

## Model training

Train the model based on your own training dataset with [model_train](https://github.com/Ryan21wy/PLM_Segformer/blob/master/training/model_train.py#L71) function.

    model_train(train_path, val_path, model_name, save_path, arg*)

*Optionnal args*
- train_path : file path of training data
- val_path: file path of vaildation data
- model_name: file name of model parameters
- save_path: file path for saving training history and model parameters

## Prediction

The segmentation mask of each damage is predicted using optional inference phase augmentation methods with [model_prediction](https://github.com/Ryan21wy/PLM_Segformer/blob/master/inference/model_prediction.py#L17) function.

    model_prediction(img_path, model_dir, label_path=None, save_path=None, n_class=2, crop_size=None, TTA=False, TLC=False, post=False)

*Optionnal args*
- img_path: file path of PLM images
- model_dir: file path of saved model parameters
- label_path: file path of damage annotation of PLM images, which uesd to calculate the evaluation metrics
- save_path: file path for saving the segmentation mask of each damage
- n_class: num of classes, background counts as well
- crop_size: the size of image patches when using the resizing and cropping method.If none, using resized images for prediction
- TTA: If true, using Test Time Augmentation
- TLC: If true, using Test-time Local Converter method
- post: If true, using image post-processing methods

## Usage

The PLM-Segformer models are provided in [release](https://github.com/Ryan21wy/PLM_Segformer/releases/download/v1.0/PLM_Segformer_models.zip).

The example codes for usage is included in [demo.ipynb](demo.ipynb).

## Contact

Wang Yue   
E-mail: ryanwy@csu.edu.cn 
