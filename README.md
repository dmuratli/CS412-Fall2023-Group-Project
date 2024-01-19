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

The methodology employed in this project encompasses a systematic exploration of five distinct machine learning algorithms to discern the most accurate approach. To evaluate the impact of outlier treatment, various experiments were conducted, including dropping detected outliers, utilizing data imputation with KNN, and training models on both approaches. The subsequent sections detail the approach for each algorithm.

The model building process commenced with a Decision Tree. Utilizing scikit-learn in Python, a regressor was instantiated with a maximum depth of 10 and squared error as the criterion. The model underwent training on the specified data, followed by hyperparameter tuning through grid search. The resulting decision tree was visualized, and predictions were evaluated based on Mean Squared Error (MSE) and R2 score.

The approach extended to RandomForestRegressor, configuring parameters such as the number of estimators, maximum depth, and minimum samples split. A grid search optimized the model, and predictions on the test set were assessed through MSE. This process aimed to enhance the accuracy of regression predictions.

Leveraging the xgboost library, an XGBoost approach was implemented with hyperparameter tuning via GridSearchCV. The model's performance was evaluated using metrics like Mean Absolute Error (MAE), MSE, and R-squared, optimizing for accurate regression predictions.

As part of feature engineering, Clustering using the K-Means algorithm was explored. The Elbow Method guided the determination of the optimal number of clusters. K-Means clustering with the chosen number of clusters (num_clusters = 6) identified patterns within the data, and the resulting dataframe was saved for further analysis.

Two Neural Network models were constructed—model_NN_1 utilized the original feature set with Word2Vec vectors, and model_NN_2 used the normalized feature set. Both models consisted of an input layer with 128 neurons, a hidden layer with 64 neurons, and an output layer with a sigmoid activation function. Models were compiled using the Adam optimizer with a learning rate of 0.01 and binary crossentropy loss Early stopping was implemented to monitor validation loss during training. Both models were evaluated for test accuracies to explore the predictive capabilities of the dataset using different feature sets.

This methodological diversity allowed for a comprehensive evaluation of the strengths and weaknesses of each approach, leading to an informed selection of the most effective model for accurate regression predictions on the given data.
