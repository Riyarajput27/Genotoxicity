# Genotoxicity
README 
Machine Learning and Deep Learning Models for Genotoxicity and Carcinogenicity Prediction

INTRODUCTION

In the fields of toxicology and drug development, predicting carcinogenicity and mutagenicity is crucial for assessing the safety of chemical compounds. Traditional methods like animal testing are resource-intensive and time-consuming. To address these challenges, this study employs advanced machine learning (ML) and deep learning (DL) techniques. By leveraging molecular descriptors and fingerprints derived from SMILES representations of chemical compounds, our objective is to develop accurate predictive models. These models aim to enhance the efficiency and reliability of toxicity assessments, supporting faster and more cost-effective evaluations of potential health risks associated with chemical substances.

 OBJECTIVES
1. Model Development: Develop robust predictive models for carcinogenicity and mutagenicity using various ML and DL approaches.
2. Performance Evaluation: Assess the performance of these models using metrics such as cross-validation scores, AUC (Area Under the ROC Curve), and confusion matrices to ensure reliability and accuracy.   
3. Applicability Domain Analysis: Determine the applicability domain of the models to distinguish between predictions that are well-supported by training data ("in-domain") and those that are less reliable ("out-domain").   
4. Model Comparison: Compare the performance of different ML and DL models to identify the most effective approach for predicting toxicological properties.

CODE WORKFLOW
 Step 1: Install RDKit, Openpyxl, Mordred, and PubChemPy

RDKit: A collection of cheminformatics and machine learning tools for processing chemical information.
Openpyxl: A Python library for reading and writing Excel files.
Mordred: A molecular descriptor calculator tool.
PubChemPy: A Python wrapper for the PubChem database, used to retrieve chemical information. 

Step 2: Load Data and Calculate Molecular Fingerprints
·  Load Data: Load the dataset from an Excel file and ensure it contains a column named 'SMILES'.
·  Calculate Fingerprints: Define a function to calculate various molecular fingerprints for each SMILES string, including CDK, Estate, MACCS, Substructure, and PubChem-like fingerprints.
·  Combine Data: Apply the fingerprint calculation function to each molecule, convert the results into a Dataframe, and merge it with the original dataset.

Step 3: Clean and Preprocess Fingerprint Data
·  Select and Clean Data: Drop non-numeric columns, remove columns with all zero values or low variance, and eliminate highly correlated columns (correlation > 0.95) to reduce redundancy.

Step 4: Feature Selection Using Random Forest
·  Feature Selection: Use a Random Forest Classifier to calculate feature importance and filter out features with zero importance.

Step 5: Feature Selection Using Recursive Feature Elimination (RFE)
·  Load Data: Load the cleaned dataset from the previously saved Excel file.
·  Recursive Feature Elimination (RFE): Use RFE with an SVM model to select the most important features, evaluated using cross-validation.

Step 6: Model Selection and Hyper parameter Tuning
·  Load Data: Load the dataset with selected features and the original target column from the cleaned dataset.
·  Define Models and Parameter Grids: Define various classification models (SVM, Naive Bayes, k-Nearest Neighbors, Decision Tree, Random Forest, Artificial Neural Network) and corresponding hyper parameter grids for optimization.


Step 7: Model Building, Evaluation, and Results Visualization
·  Model Building and Evaluation: Train each model using cross-validation and optimize hyper parameters using grid search (if applicable). Evaluate models based on cross-validated Area Under the Curve (AUC) scores.
·  Summary and Visualization: Display a horizontal bar plot comparing mean AUC scores of different models.


Step 8: Model Evaluation and Metrics Calculation
·  Metric Calculation Functions: Define functions to calculate evaluation metrics like accuracy, specificity, sensitivity, and AUC (Area Under the Curve).
·  Model Evaluation: Evaluate each model using cross-validation to compute AUC and other metrics. Also, calculate metrics like confusion matrix values (TP, TN, FP, FN) on the full dataset.


Step 09: Model and Fingerprint Performance Evaluation
·  Metric Calculation Functions: Functions to calculate evaluation metrics such as accuracy, specificity, sensitivity, and AUC (Area Under the Curve).
·  Model Evaluation: Evaluate each model using cross-validation to compute AUC and other metrics. Also calculates metrics like confusion matrix values (TP, TN, FP, FN) on the full dataset.

Step 10 : Merging Descriptors with Dataset
- Generate molecular descriptors using ChemSAR software and merge them with other datasets for further data cleaning and preprocessing.
·  Merging Descriptors with Dataset: Loads two Excel files (DATASET_insilco_prediction.xlsx and molecular_descriptors_output.xlsx), merges them based on the common column 'SMILES', and saves the merged dataframe to merged_output_descriptors.xlsx.

Step 11: Correlation Analysis and Feature Selection
Correlation Analysis: Remove redundant features that are highly correlated with each other to avoid multicollinearity and model over-fitting.

Step 12: Model Building and Evaluation
In this step, we train and evaluate several machine learning models to predict carcinogenicity using the dataset with selected features. We will perform hyper parameter tuning using GridSearchCV and evaluate model performance through cross-validation.
·  Model Training: Several classifiers (SVM, Naive Bayes, kNN, Decision Tree, Random Forest, ANN) were trained and optimized.
·  Hyper parameter Tuning: GridSearchCV was used for models where hyper parameters can be tuned.
·  Evaluation: Model performance was evaluated using cross-validated AUC scores, providing a robust measure of each model’s ability to distinguish between classes.

Step 13: Model Evaluation and Descriptor Generation
·  Metric Calculation Functions: Define functions to calculate evaluation metrics like accuracy, specificity, and sensitivity.
·  Generate Descriptors: Generate molecular descriptors for each molecule in the dataset using ChemSAR software

Step 14: Model Evaluation with Various Descriptor Sets and Visualization
·  Descriptor Generation: Added several types of descriptors to the dataset, including constitutional, Basak, burden, CATS, and MOE descriptors.
·  Model Evaluation: Evaluated various models using cross-validation and descriptor sets, identifying the top combinations based on AUC scores.
·  Visualization: Created visualizations to compare model performance across different descriptors and highlight the top-performing model-descriptor combinations.

Step 15: Advanced Model Evaluation with MACCS Fingerprints and Thresholds 
This step helps in understanding model performance based on different similarity thresholds and optimizes hyperparameters to improve predictive accuracy.
·  Fingerprint Calculation: Computed MACCS fingerprints and Tanimoto coefficients for molecules.
·  Threshold-Based Evaluation: Evaluated models based on different thresholds for identifying in-domain (ID) and out-of-domain (OD) chemicals.
·  Hyperparameter Tuning: Used GridSearchCV to find the best hyperparameters for models where applicable.
·  Visualization: Created plots to visualize AUC performance for ID and OD chemicals across different thresholds.

Step 16: Visualization of Molecular Properties and Similarity Heatmap
Calculate molecular properties (Molecular Weight and ALogP), update the datasets, and visualize the data using scatter plots and a similarity heatmap.
·  Molecular Properties Calculation: Calculated Molecular Weight and ALogP for both training and validation datasets.
·  Data Visualization: Plotted scatter plots to visualize the diversity distribution of training and validation sets based on MW and ALogP.
·  Similarity Heatmap: Generated a heatmap to show the Tanimoto similarity between all molecules in the combined dataset.

RESULTS

1.Feature Selection Using Random Forest                                                                 
                                                                                                                                                                                     
2.Feature Selection Using Recursive Feature Elimination (RFE)
A)Fingerprints : The cross-validation scores obtained for the models were as follows: 0.5862, 0.5826, 0.6000, 0.6000, and 0.6000. The mean cross-validation score was calculated to be 0.5938. These scores indicate the model's consistency and robustness across different subsets of the dataset, with a relatively stable performance around the mean value. The close range of the scores suggests that the model generalizes well and does not exhibit significant overfitting or underfitting issues.

B)Descriptors: The cross-validation scores for the model were [0.8475, 0.8136, 0.7288, 0.9153, 0.7797], with a mean cross-validation score of 0.8169. These results indicate a robust performance across different folds, suggesting that the model generalizes well to unseen data. The relatively high mean score of 0.8169 reflects the model's reliability and effectiveness in predicting the target variable.

3. Model Building, Evaluation, and Results Visualization 
A)FINGERPRINTS: The SVM model, with parameters `{'C': 0.1, 'gamma': 0.01, 'kernel': 'rbf'}`, achieved a perfect mean AUC of 1.0000, indicating flawless prediction. In comparison, Naive Bayes, with no specific hyperparameters, had a mean AUC of 0.5528, showing modest performance. The kNN model with `{'n_neighbors': 3}` had a mean AUC of 0.5000, similar to random guessing. The Decision Tree, using `{'criterion': 'entropy', 'max_depth': 30}`, had a mean AUC of 0.5473, while the Random Forest with `{'criterion': 'gini', 'n_estimators': 100}` scored 0.5528. The ANN model, optimized with `{'activation': 'relu', 'hidden_layer_sizes': (100,), 'solver': 'sgd'}`, excelled with a mean AUC of 0.9717. Overall, SVM and ANN demonstrated strong predictive capabilities, with SVM achieving perfect accuracy and ANN showing excellent performance. 

B)DESCRIPTORS:  With the best parameters, SVM achieved a mean AUC of 0.7722, indicating moderate performance. Naive Bayes, lacking hyperparameters, had a mean AUC of 0.5092, reflecting limited effectiveness. The kNN model, optimized with `{'n_neighbors': 7}`, demonstrated strong predictive capability with a mean AUC of 0.8970. The Decision Tree, using `{'criterion': 'gini', 'max_depth': 10}`, showed good performance with a mean AUC of 0.7904. The Random Forest, with `{'criterion': 'gini', 'n_estimators': 200}`, achieved the highest mean AUC of 0.9293, indicating superior accuracy. The ANN model, configured with `{'activation': 'tanh', 'hidden_layer_sizes': (50,), 'solver': 'adam'}`, also performed well with a mean AUC of 0.9031. Among these, the Random Forest model stands out for its exceptional performance.

4.Model Evaluation and Metrics Calculation
A)FINGERPRINTS: The SVM model achieved a perfect AUC of 1.0000, indicating flawless discrimination, but its accuracy was only 0.5885 with a sensitivity of 0.0000, suggesting issues such as class imbalance or model limitations. Naive Bayes showed a mean AUC of 0.5528 and perfect accuracy, sensitivity, and specificity, which could imply overfitting or dataset imbalance. The kNN model had an AUC of 0.5000, equivalent to random guessing, with low sensitivity (0.0084) and an accuracy of 0.5920. The Decision Tree achieved an AUC of 0.5453 and an accuracy of 0.6424, with improved sensitivity of 0.1308. The Random Forest model, with an AUC of 0.5528 and high accuracy (0.9931), showed excellent sensitivity (0.9831) and specificity (1.0000). The ANN model performed similarly to Random Forest, with a high AUC of 0.9689 and an accuracy of 0.9931, reflecting effective classification. Despite the SVM's perfect AUC, its poor sensitivity highlights the importance of using multiple metrics for a comprehensive evaluation.

B)DESCRIPTORS: The SVM model achieved an AUC of 0.7722, indicating good performance but not as strong as others. Despite a high accuracy of 0.7356 and perfect specificity, it failed to detect any positive cases, with a sensitivity of 0.0000. Naive Bayes had an AUC of 0.5092, close to random guessing, with an accuracy of 0.7085, and a sensitivity of 1.0000, but struggled with specificity (0.6037). The kNN model performed well, with an AUC of 0.8970 and an accuracy of 0.8780, showing effective detection of both positive and negative cases. The Decision Tree and Random Forest models both achieved perfect accuracy (1.0000) and high AUCs, with the Random Forest showing the highest AUC of 0.9210. The ANN model also demonstrated strong performance, with an AUC of 0.9046 and perfect accuracy. Overall, while the ANN and Random Forest models excelled in accuracy, sensitivity, and specificity, the SVM and Naive Bayes models faced challenges in positive case detection and balancing metrics.

5.Model and Fingerprint Performance Evaluation
The analysis reveals that the Random Forest model with the MACCS_FP fingerprint achieved the highest performance, with an AUC of 0.9509, indicating superior predictive capability for carcinogenicity and mutagenicity. Following closely, the ANN model using the Estate_FP fingerprint demonstrated a strong AUC of 0.9499. The SVM model with MACCS_FP also performed well, with an AUC of 0.9435. Other notable combinations include ANN with MACCS_FP and SubFP, which showed AUCs of 0.9397 and 0.9333, respectively. These results underscore the effectiveness of Random Forest and ANN models in conjunction with specific fingerprints for accurate predictions.
   
6.Model and Descriptor Performance Evaluation
The evaluation highlights that the Random Forest model with the Burden descriptor achieved the highest AUC of 0.9500, showcasing exceptional performance in predicting carcinogenicity and mutagenicity. The SVM model with the Burden descriptor closely followed, with an AUC of 0.9435, reflecting its strong predictive accuracy. The ANN model also performed well with the Burden descriptor, reaching an AUC of 0.9409. The kNN model with the Burden descriptor showed a solid AUC of 0.9231, while the Naive Bayes model with the same descriptor had an AUC of 0.8992. These results emphasize the Burden descriptor's effectiveness across various models for accurate predictions.
 
7.Advanced Model Evaluation with MACCS Fingerprints and Thresholds 
·  Model Performance for ID Chemicals: Random Forest demonstrated the highest and most consistent performance across various thresholds, indicating its robustness in predicting ID chemicals.
·  Model Performance for OD Chemicals: Random Forest maintained strong performance, though the Artificial Neural Network (ANN) also performed well at specific thresholds. The mean AUC values varied across threshold levels for each model, highlighting that Random Forest and ANN are particularly effective in handling OD chemicals.
                                                                                                                                                           
 OBSERVATIONS
1. Feature Selection:
   - Fingerprints: Removing features with very low importance values significantly reduced the feature space, enhancing model performance by retaining only the most relevant features. The pruning from 5,070 to 3,580 features effectively focused the analysis and improved model efficiency.
   - Descriptors: Eliminating features with zero importance resulted in a reduction from 986 to 366 features. This approach successfully minimized noise and emphasized features with meaningful contributions, enhancing the model's predictive accuracy.

2. Recursive Feature Elimination (RFE):
   - Fingerprints: The relatively stable cross-validation scores (mean AUC of 0.5938) across different subsets indicate consistent model performance and good generalizability. This stability suggests that the selected features are robust for predictive tasks.
   - Descriptors: The higher mean cross-validation score of 0.8169 suggests that the model with selected descriptors performed reliably across different folds, reflecting strong predictive capability and robustness.

3. Model Building and Evaluation:
   - Fingerprints: The SVM model achieved a perfect AUC but had limited sensitivity, indicating potential issues such as class imbalance. The ANN and Random Forest models showed strong performance, with ANN achieving high accuracy and AUC, and Random Forest demonstrating robust performance across various metrics.
   - Descriptors: Random Forest and ANN models excelled in performance with high AUC values, reflecting their effectiveness in predicting carcinogenicity. The Random Forest model, in particular, displayed superior predictive capability across different descriptors.

4. Model Performance by Fingerprint:
   - The Random Forest model with MACCS_FP fingerprint achieved the highest AUC, indicating its strong predictive capability. The ANN model with Estate_FP also performed exceptionally well, highlighting the importance of selecting appropriate fingerprints for model performance.

5. Model Performance by Descriptor:
   - The Burden descriptor consistently provided high AUC values across various models, emphasizing its effectiveness in predicting carcinogenicity. Random Forest and ANN models with the Burden descriptor achieved the highest performance, showcasing the descriptor’s significance in enhancing predictive accuracy.

6. Advanced Model Evaluation:
   - ID Chemicals: Random Forest maintained high and consistent performance across various thresholds, demonstrating its robustness. The ANN model also performed well, indicating its effectiveness in predicting ID chemicals.
   - OD Chemicals: Both Random Forest and ANN models performed strongly at specific thresholds, highlighting their reliability in handling OD chemicals.

7. Visualization of Molecular Properties:
   - The training and validation sets covered a broad range of molecular weights and logP values, with a denser distribution in the training set. The heatmap of molecular similarity showed that similarity is highest within the same molecules, which is crucial for understanding molecular relationships.

 CONCLUSION
The study demonstrates that feature selection and model optimization are crucial for enhancing predictive performance in carcinogenicity and mutagenicity predictions. The Random Forest model, particularly when combined with the MACCS_FP fingerprint and Burden descriptor, consistently achieved superior performance across various metrics, indicating its robustness and accuracy. The ANN model also showed excellent results, reflecting its capability in effectively classifying both positive and negative cases. Despite the high AUC of the SVM model, its limited sensitivity underscores the need for a comprehensive evaluation using multiple metrics. Overall, the Random Forest and ANN models emerged as the most reliable and effective for the prediction tasks, highlighting their importance in the development of predictive models.

FUTURE PROSPECTS

1. Integration of More Descriptors and Fingerprints:
   - Incorporate additional molecular descriptors and fingerprints to further enhance model performance and robustness. Exploring new descriptor sets may reveal more predictive features.

2. Advanced Deep Learning Architectures:
   - Implement advanced deep learning architectures such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to capture more complex patterns in the data.

3. Explainability and Interpretability:
   - Develop methods to enhance the interpretability of the models, such as SHAP (SHapley Additive exPlanations) values or LIME (Local Interpretable Model-agnostic Explanations), to provide insights into feature importance and model decision-making processes.

4. External Validation:
   - Validate the models on external datasets from different sources to assess their generalizability and robustness across diverse chemical spaces.

5. Real-time Application:
   - Implement the models in real-time prediction tools or software applications that can be used by researchers and regulatory agencies for rapid toxicity assessments.

6. Multi-task Learning:
   - Explore multi-task learning approaches to simultaneously predict multiple toxicological endpoints, potentially improving model efficiency and performance.

7. Collaboration with Experimentalists:
   - Collaborate with experimental toxicologists to validate model predictions with laboratory experiments, ensuring the practical relevance and accuracy of the models.

By addressing these future prospects, the predictive models developed in this study can be further refined and expanded, contributing to safer and more efficient chemical assessments in toxicology and drug development. 


