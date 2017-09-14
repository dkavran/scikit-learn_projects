# scikit-learn-MNIST_SVM_one-vs-all
One-Vs-All multiclass SVM classification of MNIST digit dataset

- 10 binary SVM classificators trained with hyperparameters optimized for distinguishing all 10 digits in MNIST dataset
- Selecting predictions based on highest probability, calculated by individual classificator, for each test case
- Reaching around 0.86 F1 score
