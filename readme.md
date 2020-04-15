# Easymile, fashion mnist classification project

## 1 Installation:

### Option 1)
Create a connda environment from the environment listing file `easymile-env.txt` provided in this project:
```
conda create --name easymile2 --file easymile-env.txt
```
 
### Option 2)
Create new environment using conda, and install the packages manually using `conda install` (or `pip install` if not in a conda environment):
```
conda create --name easymile python=3
conda install numpy tensorflow keras scikit-learn
```

## 2 Project Structure

The project is using a pre-downloaded fashion mnist dataset located in the `data/fashion` directory.
It is possible to use `keras.datasets.fashion_mnist.load_data()` but it takes a long time for the download.

All the script outputs are saved in the `output/` directory, and the appropriate file extensions are automatically used.
For example, saving the model under the name `trained` will automatically generate the files:
```
output/trained.h5
output/trained.json
```

The executable files are located in the root directory:
``` 
lib_easymile.py
train.py
predict.py
```

## 3 Script usage

`train.py` is used to train and save a model. Optionally, it can load a partially trained model and continue the training. 
By default it trains a new model, for 30 epochs, and saves it under the `trained` name.
Usage:
```
python train.py [-s <trained_model_savename>] [-l <pretrained_model_loadname>] [-e EPOCHS]
```

`predict.py` is used to load a pretrained model and predict a set of `x` samples. The set must be preprocessed (scaled to [0..1]). 
By default it uses the `_best_trained` saved model and it predicts the `x_test` from fashion mnist. 
Use the preprocessed sampled set `x_to_predict` or  use test code inside the script to manually generate new preprocessed sets to predict. 
Usage: 
```
python predict.py [-h] [-m <trained_model_savename>] [-d <dataset with preprocessed x samples>] [-s SAVEFILE]
```

`libeasymile` contains functions implementing the base project functionalities:
- load data set
- preprocess dataset
- create FC model
- create MiniVGGNet model
- train model
- save model
- load model
- evaluate model
- predict
- K cross validation
- multi train and save best

## 4 Research notes

The project uses the **Keras** on top of **Tensorflow** instead of using Tensorflow directly. This allows for faster research and development.
The K fold cross validation accuracy is:
```
Accuracy: 0.9056 (+- 0.0032)
```

### Model Choices
The basic FC model is extremely fast to design and offers a starting accuracy of ~~ `0.865`
 
For current model used,  `MinVGGNet`, several network designs where tested to arrive to the following configuration:
- 3 FC layers give a good trade-of between high accuracy and fast training
- pyramidal FC sizes are as good or slightly better than constant size FC, and have faster training
- optimal number of pooling layers is 2
- having 2 conv + relu per pooling is better than 1 conv + relu per pooling
- dropout layers can help with overfitting but give lower accuracy and longer train times
- Batch normalization does not improve performance
- S2 regularisation with `0.001` efficiently handles overfitting

### Overfitting and stopping conditions

Dropout layers, S1 and S2 regularisation where tested.
Dropout and S1 are suboptimal for this project, S2 efficiently handles overfitting

A slight drop in test accuracy can be observed over 40-50 epochs. The effect is too small to address at this project stage.
In the future a sliding window over can be implemented over the epochs training to capture the most recent model and their accuracies. 
Once the average loss of training accuracy overpasses a limit, the training can be stoped and the best model in the sliding window will be selected.

### Data preprocessing
The data is already standardised (28,28) size gray images so at this stage no further preprocessing is required above a simple MinMax scaling
In the future we can apply standard preprocessing (mean/std subtraction/division) and augmentation (random crops + flips)

### Hyperparamenters
Relu activation layers, RMSProp optimizer, Categorical Crossentropy loss, Accuracy Metrics - are the most used hyperparameters for this types of projects.
More exploration can be done in the future:
- Optimizers: Adam and SGD
- Activation functions: Sigmoid, Tanh, ELU and Leaky RELU
- Loss functions: Sparse Categorical Crossentropy, Kullback Leibler Divergence
- Performance metrics: log loss, Multiclass Confusion Matrix (Precision, Recall, F1 Score), Multi-class ROC  
  

 