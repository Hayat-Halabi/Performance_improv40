# Performance_improv40
# Monitoring Jungle Health in National Parks
# Scenario:
In the United States, the well-being of our national park is facing growing threats, including climate change, deforestation, and human activities. Safeguarding these vital ecosystems for future generations' benefit is of utmost importance to the National Park Service.

# Objective:
Your objective is to develop a deep learning model that assesses the health of jungles in various national parks across the USA. The health status of a jungle is represented as a binary value: 1 for a healthy jungle and 0 for an unhealthy jungle.

# Problem Statement:
To construct a robust model, it is recommended that you implement hyperparameter tuning and incorporate dropout in your neural network architecture. This will help ensure the model's accuracy and generalizability, enabling the National Park Service to effectively monitor and conserve these invaluable ecosystems.

### Dataset Columns:
Park Name: Name of the national park

Average Temperature: Average annual temperature of the park in Celsius

Rainfall: Annual rainfall in mm.

Human Intervention: Number of human-made constructions or interventions in the park (e.g., roads, buildings)

Wildlife Population: Number of wild animals spotted in a year in the park

Vegetation Density: Percentage of area covered by vegetation.

Air Quality Index: Air Quality Index of the Park

Water Quality: Quality of water sources in the park on a scale of 1–10 (10 being the cleanest)

Jungle Health: 1 if the jungle is healthy, 0 otherwise.

# Directions:
Install keras-tuner with the pip install command.
Import the required libraries.
Data Preprocessing:
Load the dataset from the specified CSV (Comma Separated Values) file using Pandas.
Preprocess the data by converting the categorical variable Park_Name into one-hot encoded columns.
Data Splitting:
Split the data into features (X) and the target variable (y).
Split the data into training and test sets using train_test_split from Scikit-Learn.
Define the model-building function:
Define a function named build_model taking a hyperparameter (hp) as an argument.
Create a Sequential model for building a neural network.
Add an input layer based on the number of features in the training data.
Start a loop to define hidden layers.
Add a dense layer with variable units and ReLU activation in each iteration of the loop.
Add a dropout layer with a variable dropout rate after each dense layer.
Add an output layer with a single neuron and sigmoid activation for binary classification.
Compile the model with a variable learning rate, binary cross-entropy loss, and accuracy metric.
Return the compiled neural network model.
Hyperparameter Tuning:
Create a Keras Tuner instance (tuner) and specify the search space for hyperparameters.
Use tuner.search to perform hyperparameter tuning with cross-validation. Ensure that it runs without errors.
Get the best hyperparameters:
Retrieve the best hyperparameters using tuner.get_best_hyperparameters. This will give you the optimal hyperparameters for your model.
Model Training:
Build a new model with the best hyperparameters using tuner.hypermodel.build.
Model Evaluation:
Finally, evaluate the trained model on your test data using the model.evaluate.
Print Test Accuracy:
Print the test accuracy to see how well your model performs on unseen data.
```python
# pip install keras-tuner
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
data_path = 'jungle_health_data.csv'
df = pd.read_csv(data_path)
# Convert Park_Name to one-hot encoded columns
df = pd.get_dummies(df, columns=['Park_Name'], drop_first=True, dtype=float)

# Split data into features and target
X = df.drop('Jungle_Health', axis=1)
y = df['Jungle_Health']

# 3. Splitting the data into training and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X.info()
# Define a function to build a neural network model with hyperparameters
def build_model(hp):
    # Create a Sequential model (a linear stack of layers)
    model = keras.Sequential()

    # Input layer
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))

    # Hidden layers: The number of hidden layers and their properties are determined by a hyperparameter search.
    for i in range(hp.Int('num_layers', 1, 5)):  # Iterate over a range of possible numbers of hidden layers (1 to 5).
        # Add a dense (fully connected) layer with variable units and ReLU activation.
        model.add(layers.Dense(units=hp.Int('units_' + str(i), 32, 256, 32), activation='relu'))

        # Add a dropout layer with variable dropout rate.
        model.add(layers.Dropout(rate=hp.Float('dropout_' + str(i), 0.0, 0.5, step=0.1)))

    # Output layer: A single neuron with sigmoid activation for binary classification.
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model:
    # Use the Adam optimizer with a variable learning rate.
    # Binary cross-entropy is used as the loss function for binary classification.
    # Accuracy is used as a metric for evaluation.
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),  # Learning rate is a hyperparameter choice.
        loss='binary_crossentropy',  # Binary cross-entropy loss for binary classification.
        metrics=['accuracy'])  # Model performance metric.

    return model  # Return the compiled model.
```
Hyperparameter Tuning:
Import the necessary Keras Tuner module.
Define the tuner with hyperparameter tuning configuration.
Summarize the search space, showing the range of hyperparameters to explore.
Perform the hyperparameter search by training the models.
Retrieve the optimal hyperparameters from the search results (best combination).
```python
# Import the necessary Keras Tuner module
from kerastuner.tuners import RandomSearch

# Define the tuner with hyperparameter tuning configuration
tuner = RandomSearch(
    build_model,                # The build_model function defined earlier.
    objective='val_accuracy',   # Objective to optimize (maximize validation accuracy).
    max_trials=5,               # Maximum number of trials (combinations of hyperparameters) to run.
    executions_per_trial=3,     # Number of executions (training runs) per trial to reduce variance.
    directory='tuning_dir',     # Directory to store tuning results and checkpoints.
    project_name='jungle_health_tuning'  # Name for this hyperparameter tuning project.
)

# Summarize the search space, showing the range of hyperparameters to explore.
tuner.search_space_summary()

# Perform the hyperparameter search by training the models
tuner.search(X_train, y_train, epochs=30, validation_split=0.2)

# Retrieve the optimal hyperparameters from the search results (best combination)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build a model using the best hyperparameters found during the hyperparameter tuning.
model = tuner.hypermodel.build(best_hps)

# Train the model on the training data.
# - X_train and y_train are the training data and labels.
# - epochs: Number of training iterations.
# - validation_split: Fraction of training data to use for validation.
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Evaluate the trained model on the test dataset to assess its performance.

# Use the model's evaluate method to calculate the test loss and test accuracy.
# - X_test and y_test are the test data and labels.
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Print the test accuracy as a percentage.
# - test_accuracy is a decimal number, so it's converted to a percentage using formatting.
print(f"Test accuracy: {test_accuracy*100:.2f}%")


