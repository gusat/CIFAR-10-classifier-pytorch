# CIFAR-10-classifier-pytorch
learning intro to CNNs and Pytorch
live code w/ GPU at https://www.kaggle.com/mgusat/cifar-10-v0-9

```markdown
# CIFAR-10 Image Classification

This repository contains the code for a CIFAR-10 image classification project. The goal of this project is to train a deep learning model to classify images from the CIFAR-10 dataset into 10 different classes.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Steps

### 1. Data Preprocessing

- Data was preprocessed using PyTorch's data augmentation and normalization techniques.
- The dataset was split into training, validation, and test sets.

### 2. Model Architecture

- A convolutional neural network (CNN) was designed using PyTorch's `nn` module.
- The architecture consisted of multiple convolutional layers, followed by max-pooling and fully connected layers.
- Different activation functions and dropout layers were used to prevent overfitting.

### 3. Training

- The model was trained using the SGD and Adam optimizers and a cross-entropy loss function.
- Training was performed on a GPU for faster computation.
- The training loss and validation accuracy were recorded for each epoch.

### 4. Evaluation

- The trained model was evaluated on the validation set to measure its performance.
- The accuracy and loss were calculated on the validation set.

### 5. Testing

- The final trained model was tested on a separate test set to assess its real-world performance.
- The percentage of correct predictions was computed using the highest probability prediction.

### 6. Conclusion

- The achieved validation accuracy was 71.8%, indicating that a basic model and techniques were sufficient for this task.
- Recommendations were made based on the evaluation, suggesting that further improvements are possible with advanced techniques and hyperparameter tuning.

## Usage

1. Clone the repository:

```
git clone https://github.com/gusat/CIFAR-10-classifier-pytorch.git
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Run the training script:

```
python train.py
```

4. Test the model on the test set:

```
python test.py
```

## Results

The detailed results of the training and testing can be found in the Jupyter notebook  https://www.kaggle.com/mgusat/cifar-10-v0-9

## Acknowledgments

- The project was inspired by the CIFAR-10 dataset provided by the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) website.
- PyTorch was used as the deep learning framework.
```

