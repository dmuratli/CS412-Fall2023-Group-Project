# Overview of the Repository

This repository encompasses the comprehensive implementation of our machine learning project, focusing on enhancing the understanding of homework grades based on ChatGPT prompts. Below is a succinct breakdown of the main script and code components embedded within the repository.

### Data Preprocessing
The initial phase involves data preprocessing, including NLTK resource download and HTML text cleaning, extracting, and storing preprocessed conversations. The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/558bb9997b7da86e6e039f5162f6c35964468c12/notebook.py#L51C1-L114C20).

### Matching Prompts with Questions 
In this step, user prompts are matched with their respective codes, preprocess a set of questions, and prepare them for subsequent analysis. The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L117C1-L209C1).

### TF-IDF Vectorization 
The subsequent stage involves TF-IDF vectorization on user prompts and questions, calculating cosine similarity, and organizing results into a dataframe (question_mapping_scores). The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L211C1-L269C1).

### Outlier Treatment
To ensure data integrity, the code identifies and handles outliers by filtering entries based on average distances and a specified threshold. The codes of dropped entries are printed for reference. The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L271C1-L304C42).

### Word2Vec Vectorization
Utilizing Word2Vec, this step creates embeddings for prompts and questions, calculates cosine similarity scores, and generates individual Word2Vec models for each student. The resulting dataframe (code2word2vec) is printed for reference. The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L345C1-L401C1).

### Normalization
The normalization phase involves reading and preprocessing scores from a CSV file, removing grades received by outlier students, and performing Min-Max scaling on the grades. The resulting dataframe (normalized_scores) is printed for reference. The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L403C1-L442C25).

### Feature Engineering
Feature engineering includes sentiment analysis, keyword counting, and context detection for top-scoring students. Various scenarios are considered, and merged dataframes are created for regular scores, separate Word2Vec models, normalized scores, and separate Word2Vec models with normalized scores. The sorted dataframes are printed for reference. The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L445C1-L634C62).

### Model Training
Finally, model training is initiated, encompassing Neural Networks, Clustering, Decision Tree, RandomForestRegressor, and XGBoost. The models' Mean Squared Error (MSE) scores are evaluated to identify the optimal one.

# Methodology

The methodology employed in this project encompasses a systematic exploration of five distinct machine learning algorithms to discern the most accurate approach. To evaluate the impact of outlier treatment, various experiments were conducted, including dropping detected outliers, utilizing data imputation with KNN, and training models on both approaches.

The model building process commenced with a [Decision Tree](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L780C1-L843C37). Utilizing scikit-learn in Python, a regressor was instantiated with a maximum depth of 10 and squared error as the criterion. The model underwent training on the specified data, followed by hyperparameter tuning through grid search. The resulting decision tree was visualized, and predictions were evaluated based on Mean Squared Error (MSE) and R2 score.

The approach extended to [RandomForestRegressor](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L846C1-L898C33), configuring parameters such as the number of estimators, maximum depth, and minimum samples split. A grid search optimized the model, and predictions on the test set were assessed through MSE. This process aimed to enhance the accuracy of regression predictions.

Leveraging the [xgboost](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L901C1-L1020C1) library, an XGBoost approach was implemented with hyperparameter tuning via GridSearchCV. The model's performance was evaluated using metrics like Mean Absolute Error (MAE), MSE, and R-squared, optimizing for accurate regression predictions.

As part of feature engineering, [Clustering](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L743C1-L777C41) using the K-Means algorithm was explored. The Elbow Method guided the determination of the optimal number of clusters. K-Means clustering with the chosen number of clusters (num_clusters = 6) identified patterns within the data, and the resulting dataframe was saved for further analysis.

Two [Neural Network models](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L661C1-L741C1) were constructed, model_NN_1 utilized the original feature set with Word2Vec vectors, and model_NN_2 used the normalized feature set. Both models consisted of an input layer with 128 neurons, a hidden layer with 64 neurons, and an output layer with a sigmoid activation function. Models were compiled using the Adam optimizer with a learning rate of 0.01 and binary crossentropy loss Early stopping was implemented to monitor validation loss during training. Both models were evaluated for test accuracies to explore the predictive capabilities of the dataset using different feature sets.

This methodological diversity allowed for a comprehensive evaluation of the strengths and weaknesses of each approach, leading to an informed selection of the most effective model for accurate regression predictions on the given data.

# Results

## Clustering:
The clustering plot before (Figure 2) and after kNN imputation (Figure 3) displays the distribution of grades in a 2-dimensional space. Each dot represents a conversation, color-coded by the cluster it belongs to. There appears to be a degree of separation between clusters, suggesting that the conversations have some underlying patterns that are related to the grades received. The clustering does not show distinct groupings by grades in either case, indicating that the relationship between conversation features and grades is complex.

![Figure 2: Results of clustering before kNN imputation (outliers and students with empty prompts are dropped)](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/clustering.png?raw=true)
*Caption: Results of clustering before kNN imputation (outliers and students with empty prompts are dropped)*

![Figure 3: Results of clustering after kNN imputation](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/clustering_knn.png?raw=true)

## MAE and MSE Results:
The MAE and MSE plots before (Figure 4 and 5 respectively) and after kNN imputation (Figure 6 and 7 respectively) compare the performance of different models. These include two neural network configurations (with and without word2vec vectors), Decision Tree, Random Forest, XGBoost, and two dummy classifiers (mean and median strategies).

![Figure 4: MAEs before kNN imputation (outliers and students with empty prompts are dropped)](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/mae.png?raw=true)
*Caption: MAEs before kNN imputation (outliers and students with empty prompts are dropped)*

![Figure 5: MSEs before kNN imputation (outliers and students with empty prompts are dropped)](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/mse.png?raw=true)
*Caption: MSEs before kNN imputation (outliers and students with empty prompts are dropped)*

![Figure 6: MAEs after kNN imputation](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/mae_knn.png?raw=true)
*Caption: MAEs after kNN imputation*

![Figure 7: MSEs after kNN imputation](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/mse_knn.png?raw=true)
*Caption: MSEs after kNN imputation*

### Neural Networks:
- Contrary to initial expectations, the neural network model that did not include the word2vec vectors performed better, with lower MAE and MSE values. This may indicate that in this specific context, the word2vec vectors did not add valuable information and perhaps introduced noise or overfitting.
- After kNN imputation, the error rates for both neural network configurations increased, reaffirming that the imputation method may not be well-suited to this dataset.

### Decision Tree:
- The Decision Tree shows a notable disparity between training and test errors, with a significantly lower error on the training set. This is indicative of overfitting, where the model captures noise in the training data that does not generalize well to unseen data.
- Post-kNN imputation, the Decision Tree's performance deteriorates further on the test set, suggesting that the imputation might have introduced complexities that the model is overfitting to even more.

[Figure 8](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/hw.pdf) displays the decision tree in a graphic format.

### Random Forest:
- The Random Forest model, which is an ensemble of decision trees, typically reduces overfitting by averaging the results of individual trees. This is observed in the less pronounced gap between training and test errors compared to the single Decision Tree model.
- After kNN imputation, the Random Forest's errors increased, but it still maintained a more stable performance relative to the Decision Tree, which points to its robustness despite the potentially noisy imputed data.

### XGBoost:
- XGBoost, another ensemble method that builds trees in a sequential manner to correct the errors of the previous trees, usually performs well on structured data. The error rates pre-imputation were competitive, but there is still evidence of overfitting, as seen in the lower training error compared to test error.
- The increase in error rates after kNN imputation suggests that the sequential improvement strategy of XGBoost might be amplifying the noise introduced by the imputed data, leading to poorer generalization on the test set.

### Dummy Classifiers:
- The performance of the dummy classifiers is intriguing. These classifiers do not learn from the data; they simply predict the mean or median value of the training set grades. The fact that they perform better than most of the models on the test set is a strong signal.
- This outcome could suggest several things: the features may not be predictive enough, the models might be too complex and overfitting to the training data, or the kNN imputation could have introduced misleading information, leading to a degradation in model performance.
- The dummy classifiers serving as a better predictor than more sophisticated models on the test set is a critical finding. It implies that most of the models (with the neural networks being the only exception) are not capturing the underlying patterns in the data effectively and are instead learning from noise. 

In conclusion, the results suggest that the neural network models, particularly when not augmented with word2vec vectors, outperform the other models in predicting homework grades from ChatGPT conversations. It is surprising that kNN imputation seems to have an adverse effect on the performance of all models, as evidenced by increased error rates post-imputation. This could imply that the missing data handled by kNN imputation introduced noise or that the patterns in the imputed data do not align well with the underlying structure of the dataset. It might be beneficial to explore alternative imputation methods or to consider feature selection to improve model performance.

