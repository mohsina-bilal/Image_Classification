# Image Classification

This repository contains code for an image classification model using TensorFlow. The model is trained to classify images into different categories such as sea, mountains, city, and forest.

## Dataset

The dataset used for training and testing the model consists of images from various categories. The images are stored in separate directories based on their respective categories.

## Model Architecture

The model architecture used for image classification is a Convolutional Neural Network (CNN). It consists of three convolutional layers with max-pooling and a fully connected layer. Dropout and data augmentation techniques are applied to reduce overfitting.

## Training and Evaluation

The model is trained on the dataset using the Adam optimizer and Sparse Categorical Crossentropy loss. It is trained for a specified number of epochs, and the training and validation accuracy and loss are monitored.

## Results

The training and validation accuracy and loss are plotted to visualize the model's performance. Additionally, the model is evaluated on new data and the predicted classes along with confidence scores are displayed.

## Usage

To use this code, follow these steps:

1. Clone the repository: `git clone https://github.com/mohsina-bilal/image-classification.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download the dataset and place it in the appropriate directory.
4. Run the script: `python image_classification.py`

## Contributing

Contributions to this project are welcome. Feel free to open issues and submit pull requests to suggest improvements or add new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
