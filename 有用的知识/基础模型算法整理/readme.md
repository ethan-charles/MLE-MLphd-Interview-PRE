# Linear Regression

1. Initialize weights w and bias b  
2. Set learning rate α  
3. For each epoch:  
   - For each data point (x_i, y_i):  
       - Predict y_pred = w * x_i + b  
       - Calculate error = y_pred - y_i  
       - Update weights: w = w - α * error * x_i  
       - Update bias: b = b - α * error  
4. Repeat until convergence or for a set number of iterations.


# Logistic Regression

1. Initialize weights w and bias b  
2. Set learning rate α  
3. For each epoch:  
   - For each data point (x_i, y_i):  
       - Compute y_pred = sigmoid(w * x_i + b)  
       - Calculate error = y_pred - y_i  
       - Update weights: w = w - α * error * x_i  
       - Update bias: b = b - α * error  
4. Repeat until convergence or for a set number of iterations.

# Sigmoid function:
sigmoid(z) = 1 / (1 + exp(-z))


# K-Nearest Neighbors (KNN)

1. Set the number of neighbors K  
2. For each test data point:  
   - Calculate the distance between the test point and all training points  
   - Select the K nearest neighbors  
   - Assign the most common label among the K neighbors as the prediction


# Decision Tree

1. Start at the root node  
2. For each node, if all data points belong to the same class:  
   - Return the class  
3. Otherwise:  
   - Calculate the best feature to split on based on a criterion (e.g., Gini impurity, Information gain)  
   - Split the data into subsets based on the selected feature  
   - Repeat the process recursively for each subset until reaching a stopping condition (e.g., max depth, pure leaf)  
4. Assign the majority class in the leaf nodes.


# Support Vector Machine (SVM)

1. Initialize weights w and bias b  
2. Set learning rate α and regularization parameter λ  
3. For each epoch:  
   - For each data point (x_i, y_i):  
       - If y_i * (w * x_i + b) >= 1:  
           - Update weights: w = w - α * λ * w  
       - Else:  
           - Update weights: w = w - α * (λ * w - y_i * x_i)  
           - Update bias: b = b + α * y_i  
4. Repeat until convergence or for a set number of iterations.


# K-Means Clustering

1. Initialize K cluster centroids randomly  
2. For each data point:  
   - Assign the point to the nearest centroid  
3. Update centroids by calculating the mean of the points in each cluster  
4. Repeat steps 2-3 until convergence (cent


# Naive Bayes

1. For each class y in the training data:

   1.1 Compute the prior P(y) = count(y) / total_count

   1.2 For each feature x_i:
   
       - Compute the likelihood P(x_i | y)

2. For each test data point x:

   2.1 Compute the posterior for each class y:

       P(y | x) = P(y) * Π P(x_i | y)

   2.2 Assign the class with the highest posterior probability.

# Random Forest

1. For each tree in the forest:

   1.1 Select a random subset of the training data (with replacement)

   1.2 Build a decision tree using the subset

   1.3 At each node, randomly select a subset of features to split on

2. For each test data point:

   2.1 Get predictions from all trees

   2.2 Assign the majority vote as the final prediction.

# Gradient Boosting

1. Initialize the model with a constant prediction (e.g., mean of the target values)

2. For each iteration:

   2.1 Compute the residuals (errors) between the current predictions and actual target values

   2.2 Train a weak learner (e.g., a decision tree) to predict the residuals

   2.3 Update the model by adding the weak learner's prediction to the current model

   2.4 Multiply the weak learner's predictions by a learning rate to control the step size

3. Repeat until the model reaches a stopping criterion (e.g., number of iterations or convergence).

