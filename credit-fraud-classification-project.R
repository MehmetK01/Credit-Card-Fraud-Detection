# Clear the environment and console
rm(list = ls())
cat("\014")

# Load dataset
credit_data <- read.csv("/Users/mehmet/Desktop/creditcard.csv")

# Basic summary statistics and structure
dim(credit_data)          # Check dimensions of the dataset
str(credit_data)          # Structure of the data
summary(credit_data)      # Summary statistics for all columns

# Check for missing values
cat("Number of missing values:\n")
colSums(is.na(credit_data))

# View the first few rows
head(credit_data)

# Class distribution (Fraud vs Non-Fraud)
cat("Class distribution:\n")
table(credit_data$Class) # highly inbalanced

# Visualize class imbalance
library(ggplot2)
ggplot(credit_data, aes(x = factor(Class))) +
  geom_bar(fill = c("skyblue", "red")) +
  labs(title = "Class Distribution", x = "Class (0: Non-Fraud, 1: Fraud)", y = "Count") +
  theme_minimal()

# Summary statistics for transaction Amount
cat("Summary of Amount:\n")
summary(credit_data$Amount)

# Visualize Amount distribution
ggplot(credit_data, aes(x = Amount)) +
  geom_histogram(bins = 75, fill = "lightblue", color = "black") +
  labs(title = "Distribution of Transaction Amount", x = "Transaction Amount", y = "Frequency") +
  theme_minimal()

# Time feature: Distribution of transactions over time
ggplot(credit_data, aes(x = Time)) +
  geom_histogram(bins = 50, fill = "lightgreen", color = "black") +
  labs(title = "Transaction Distribution Over Time", x = "Time (seconds)", y = "Frequency") +
  theme_minimal()

# Correlation heatmap (optional: requires the corrplot package)
library(corrplot)
corr_matrix <- cor(credit_data[, -1])  # Exclude 'Time' column for correlation
corrplot(corr_matrix, method = "color", type = "upper", tl.cex = 0.6)

# Boxplot to visualize Amount for Fraud vs Non-Fraud
ggplot(credit_data, aes(x = factor(Class), y = Amount, fill = factor(Class))) +
  geom_boxplot() +
  labs(title = "Transaction Amount by Class", x = "Class (0: Non-Fraud, 1: Fraud)", y = "Amount") +
  scale_fill_manual(values = c("skyblue", "red")) +
  theme_minimal()

################################################################################

# Density plot for the 'Amount' variable
ggplot(credit_data, aes(x = log(Amount + 0.01), fill = factor(Class))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("genuine" = "skyblue", "fraud" = "red"), 
                    labels = c("0" = "Genuine", "1" = "Fraud")) +
  labs(title = "Density Plot of Transaction Amount by Class",
       x = "Log(Transaction Amount + 0.01)", 
       y = "Density", 
       fill = "Class") +
  theme_minimal()

# Density plot for V1 (First Principal Component)
ggplot(credit_data, aes(x = V1, fill = factor(Class))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("genuine" = "skyblue", "fraud" = "red"), 
                    labels = c("0" = "Genuine", "1" = "Fraud")) +
  labs(title = "Density Plot of First Principal Component (V1) by Class",
       x = "V1", 
       y = "Density", 
       fill = "Class") +
  theme_minimal()

# Density plot for V2 (Second Principal Component)
ggplot(credit_data, aes(x = V2, fill = factor(Class))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("genuine" = "skyblue", "fraud" = "red"), 
                    labels = c("0" = "Genuine", "1" = "Fraud")) +
  labs(title = "Density Plot of Second Principal Component (V2) by Class",
       x = "V2", 
       y = "Density", 
       fill = "Class") +
  theme_minimal()


################################################################################
################################### ML MODELS ##################################

# Load necessary libraries
library(caret)         # For model training and evaluation
library(pROC)          # For ROC curve analysis

# Load the library
library(ROSE)

# Install the smotefamily package
#install.packages("smotefamily")

# Load the library
library(smotefamily)

library(ggplot2)       # Visualization
library(kernlab)       # For Neural Networks (Support for caret)
library(gbm)           # For GBM


# Ensure 'Class' is a factor
credit_data$Class <- as.factor(credit_data$Class)

# Check class imbalance
table(credit_data$Class)

# Split data into training (70%) and testing (30%) sets
set.seed(123)
trainIndex <- createDataPartition(credit_data$Class, p = 0.7, list = FALSE)
train_data <- credit_data[trainIndex, ]
test_data <- credit_data[-trainIndex, ]

# Apply ROSE for class balancing
train_data_balanced <- ROSE(Class ~ ., data = train_data, seed = 123)$data
table(train_data_balanced$Class)


# Ensure Class is a factor and rename levels
train_data_balanced$Class <- factor(train_data_balanced$Class, 
                                    levels = c(0, 1), 
                                    labels = c("nonfraud", "fraud"))

# Check if the levels have been renamed
print(table(train_data_balanced$Class))

# Step 2: Define Control and Evaluation Metrics

# Cross-validation setup
control <- trainControl(method = "cv", number = 5, 
                        summaryFunction = twoClassSummary, 
                        classProbs = TRUE, 
                        verboseIter = FALSE)

# Define metrics to use
metric <- "ROC"

# Step 3: Logistic Regression
# Train Logistic Regression
set.seed(123)
logistic_model <- train(Class ~ ., data = train_data_balanced, 
                        method = "glm", 
                        family = "binomial", 
                        trControl = control,
                        metric = metric)

# Ensure Class in test_data has the same levels as train_data
test_data$Class <- factor(test_data$Class, 
                          levels = c(0, 1), 
                          labels = c("nonfraud", "fraud"))

# Verify levels
print(table(test_data$Class))



# Evaluate on test data
logistic_preds <- predict(logistic_model, test_data, type = "prob")
logistic_roc <- roc(test_data$Class, logistic_preds[,2], levels = rev(levels(test_data$Class)))
logistic_sensitivity <- sensitivity(predict(logistic_model, test_data), test_data$Class)

# Print Results
cat("Logistic Regression - ROC:", logistic_roc$auc, "Sensitivity:", logistic_sensitivity, "\n")



# Step 5: GBM (Gradient Boosting Machine)
# Train GBM Model
set.seed(123)
gbm_model <- train(Class ~ ., data = train_data_balanced, 
                   method = "gbm", 
                   trControl = control,
                   verbose = FALSE,
                   metric = metric)

# Evaluate on test data
gbm_preds <- predict(gbm_model, test_data, type = "prob")
gbm_roc <- roc(test_data$Class, gbm_preds[,2], levels = rev(levels(test_data$Class)))
gbm_sensitivity <- sensitivity(predict(gbm_model, test_data), test_data$Class)

# Print Results
cat("GBM Model - ROC:", gbm_roc$auc, "Sensitivity:", gbm_sensitivity, "\n")


# Step 6: Neural Network
# Train Neural Network Model
set.seed(123)
nnet_model <- train(Class ~ ., data = train_data_balanced, 
                    method = "nnet", 
                    trControl = control,
                    tuneLength = 5,
                    trace = FALSE,
                    metric = metric)

# Evaluate on test data
nnet_preds <- predict(nnet_model, test_data, type = "prob")
nnet_roc <- roc(test_data$Class, nnet_preds[,2], levels = rev(levels(test_data$Class)))
nnet_sensitivity <- sensitivity(predict(nnet_model, test_data), test_data$Class)

# Print Results
cat("Neural Network - ROC:", nnet_roc$auc, "Sensitivity:", nnet_sensitivity, "\n")


# Specify the file to save the output
sink("model_output-fraudClassification-project.txt")

# Step 7: Compare Model Results
# Compare ROC and Sensitivity
results <- data.frame(
  #Model = c("Logistic Regression", "KNN", "GBM", "Neural Network"),
  Model = c("Logistic Regression", "GBM", "Neural Network"),
  #ROC = c(logistic_roc$auc, knn_roc$auc, gbm_roc$auc, nnet_roc$auc),
  ROC = c(logistic_roc$auc, gbm_roc$auc, nnet_roc$auc),
  #Sensitivity = c(logistic_sensitivity, knn_sensitivity, gbm_sensitivity, nnet_sensitivity)
  Sensitivity = c(logistic_sensitivity, gbm_sensitivity, nnet_sensitivity)
)

print(results)

##################### confusion matrix and accuracy ############################
library(caret)
library(e1071)

# First model:
# Logistic Regression Confusion Matrix and Accuracy
logistic_preds <- predict(logistic_model, test_data)
logistic_cm <- confusionMatrix(logistic_preds, test_data$Class)

# Accuracy from confusion matrix
logistic_accuracy <- logistic_cm$overall['Accuracy']
cat("Logistic Regression - Accuracy:", logistic_accuracy, "\n")



# Third Model:
# GBM Confusion Matrix and Accuracy
gbm_preds <- predict(gbm_model, test_data)
gbm_cm <- confusionMatrix(gbm_preds, test_data$Class)

# Accuracy from confusion matrix
gbm_accuracy <- gbm_cm$overall['Accuracy']
cat("GBM Model - Accuracy:", gbm_accuracy, "\n")

# Fourth Model:
# Neural Network Confusion Matrix and Accuracy
nnet_preds <- predict(nnet_model, test_data)
nnet_cm <- confusionMatrix(nnet_preds, test_data$Class)

# Accuracy from confusion matrix
nnet_accuracy <- nnet_cm$overall['Accuracy']
cat("Neural Network - Accuracy:", nnet_accuracy, "\n")


###################### Comparison of the 3 models: #############################

# Initialize a data frame to store the results (excluding KNN)
model_comparison <- data.frame(
  Model = c("Logistic Regression", "GBM", "Neural Network"),
  Accuracy = numeric(3),
  Sensitivity = numeric(3),
  Specificity = numeric(3),
  stringsAsFactors = FALSE
)

# Logistic Regression
logistic_preds <- predict(logistic_model, test_data)
logistic_cm <- confusionMatrix(logistic_preds, test_data$Class)
model_comparison$Accuracy[1] <- logistic_cm$overall['Accuracy']
model_comparison$Sensitivity[1] <- logistic_cm$byClass['Sensitivity']
model_comparison$Specificity[1] <- logistic_cm$byClass['Specificity']

# GBM Model
gbm_preds <- predict(gbm_model, test_data)
gbm_cm <- confusionMatrix(gbm_preds, test_data$Class)
model_comparison$Accuracy[2] <- gbm_cm$overall['Accuracy']
model_comparison$Sensitivity[2] <- gbm_cm$byClass['Sensitivity']
model_comparison$Specificity[2] <- gbm_cm$byClass['Specificity']

# Neural Network Model
nnet_preds <- predict(nnet_model, test_data)
nnet_cm <- confusionMatrix(nnet_preds, test_data$Class)
model_comparison$Accuracy[3] <- nnet_cm$overall['Accuracy']
model_comparison$Sensitivity[3] <- nnet_cm$byClass['Sensitivity']
model_comparison$Specificity[3] <- nnet_cm$byClass['Specificity']

# Print the comparison table
print(model_comparison)


# Stop redirecting the output
sink()
