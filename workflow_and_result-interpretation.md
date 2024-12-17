# Credit-Card-Fraud-Detection
1.  and develop a fraudulent detecting algorithm l Preprocessed the data (50k observations) and resampled the imbalanced dataset
Machine Learning Analysis using the Credit Card Fraud Detection data set from Kaggle and developed a fraudulent detecting algorithm. I applied 3 different Models (Logistic Regression, GBM, Neural Networks). 

To address the imbalance existent in the data set, I performed resampling (ROSE balancing). The first part of my analysis revolves around EDA and getting the data set better. I plot different graphs (incl. box-plots and density plots) to gather information as to the distribution of fraudulent transactions.

Afterwards, in my analysis I apply Logistic Regression, GBM (Gradient Boosting Machine), and Neural Networks. I print the results in a seperate txt file. 

I have also applied some tunings to the customized loss function. 




Best Model Interpretation:

Accuracy
The accuracy values for the three models are as follows:
Logistic Regression: 0.9882
GBM: 0.9970 (highest)
Neural Network: 0.9754
While GBM achieves the highest accuracy, it is important to note that accuracy can be misleading in fraud classification. Since fraud detection involves imbalanced data, where the majority class (non-fraud) dominates, accuracy alone may not reflect true model performance.

Sensitivity (Recall for Fraud Class)
Sensitivity is crucial in fraud detection, as it measures the ability to correctly identify fraudulent transactions (minority class):
Logistic Regression: 0.9884
GBM: 0.9973 (highest)
Neural Network: 0.9756
The GBM model achieves the highest sensitivity, meaning it correctly identifies the most fraudulent cases. This is especially important in fraud detection, where missing fraudulent transactions (false negatives) can be highly costly.

Specificity
Specificity measures the ability to correctly classify genuine transactions (majority class):
Logistic Regression: 0.8571
GBM: 0.8095
Neural Network: 0.8639 (highest)
The Neural Network achieves the highest specificity, meaning it performs slightly better at reducing false positives. However, this improvement in specificity comes at the cost of lower sensitivity and accuracy compared to the GBM model.

Conclusion:
In the context of fraud classification, the GBM (Gradient Boosting Machine) stands out as the best-performing model. It achieves the highest sensitivity (0.9973), meaning it correctly identifies nearly all fraudulent cases, and also delivers the highest accuracy (0.9970), demonstrating strong overall predictive performance. While the Neural Network slightly edges out GBM in specificity, sensitivity is typically more critical in fraud detection to minimize missed fraudulent transactions. For this reason, GBM is the most suitable choice.
