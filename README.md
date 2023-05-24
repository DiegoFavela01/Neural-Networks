# Neural-Networks
## Challenge 13 - University of Berkeley Financial Technology Boot Camp
### Data Preparation
- Reading the data: The code reads the applicants_data.csv file into a Pandas DataFrame. This file contains the dataset that will be used for training and testing the neural network model.
- Dropping irrelevant columns: The code drops the "EIN" (Employer Identification Number) and "NAME" columns from the DataFrame because they are not relevant to the binary classification model. These columns do not provide useful information for predicting the success of - Alphabet Soup-funded startups.
- Encoding categorical variables: The code uses OneHotEncoder from scikit-learn to encode the categorical variables in the dataset. Categorical variables are variables that have discrete values, such as "industry" or "region". Encoding converts categorical variables into a binary representation, making them suitable for input to the neural network model.
- Concatenating encoded variables: The code concatenates the encoded variables with the original DataFrame's numerical variables using the Pandas concat() function. This step creates a new DataFrame that contains both the encoded categorical variables and the numerical variables.
- Creating features and target datasets: The code creates the features dataset (X) and the target dataset (y) from the preprocessed DataFrame. The target dataset is defined by the "IS_SUCCESSFUL" column, which indicates whether a startup is successful or not. The remaining columns in the DataFrame define the features dataset.
- Splitting into training and testing datasets: The code splits the features and target datasets into training and testing datasets. This division is necessary to evaluate the model's performance on unseen data. The training dataset is used to train the model, while the testing dataset is used to evaluate the trained model's accuracy.
- Scaling features data: The code uses scikit-learn's StandardScaler to scale the features data. Scaling ensures that all features have similar scales, which can improve the performance of the neural network model.

### Compile and Evaluate Binary Classification Model
- Designing the neural network model: The code creates a deep neural network model using TensorFlow's Keras. The number of input features, layers, and neurons on each layer are determined based on the dataset. The model architecture can be customized by adding or removing layers and adjusting the number of neurons.
- Compiling and fitting the model: The code compiles the model by specifying the loss function, optimizer, and evaluation metric. The loss function used for binary classification is binary_crossentropy. The optimizer determines how the model is updated based on the computed gradients, and the adam optimizer is commonly used. The model is fit to the training data by specifying the number of epochs, which is the number of times the model will iterate over the entire training dataset.
- Evaluating the model: The code evaluates the trained model using the test data to calculate the model's loss and accuracy. The loss value indicates how well the model predicts the target variable, while accuracy measures the percentage of correct predictions.
- Saving the model: The code saves the trained model to an HDF5 file named AlphabetSoup.h5. Saving the model allows for future use without the need to retrain it.
## Libraries and Dependencies
This code uses the following libraries:

- Pandas
- pathlib
- tensorflow
- keras
- Scikit-learn
####  To install these libraries
```bash
!pip install pandas pathlib tensorflow keras scikit-learn

```
## Files
--- The necessary file for this the applicants_data.csv found in /Resources
