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

Our methodology involved a systematic exploration of various machine learning algorithms and outlier treatment strategies to optimize performance in understanding student responses. Five distinct machine learning algorithms were implemented, each meticulously tailored to uncover the nuances of the dataset.

### Decision Tree Regression
The team commenced with a Decision Tree model, employing scikit-learn in Python. A comprehensive grid search over hyperparameters such as criteria, maximum depth, minimum samples split, and maximum leaf nodes was conducted. The resulting Decision Tree was visualized, and its performance evaluated using Mean Squared Error (MSE) and R2 score metrics.

### Random Forest Regression
Following the Decision Tree, a RandomForestRegressor was implemented with scikit-learn. A grid search over key parameters, including the number of estimators, maximum depth, minimum samples split, and minimum samples leaf, was executed. The optimized model's performance was assessed through MSE, aiming for accurate regression predictions.

### XGBoost Regression
The XGBoost approach was adopted for regression, leveraging the xgboost library. A GridSearchCV optimized hyperparameters such as min_child_weight, gamma, subsample, colsample_bytree, and max_depth. Performance evaluation encompassed metrics like Mean Absolute Error (MAE), MSE, and R-squared.

### Clustering with K-Means
Incorporating clustering into feature engineering, the team utilized the K-Means algorithm. The Elbow Method determined the optimal number of clusters (k), and K-Means clustering was applied to unveil patterns and groupings within the data.

### Neural Network Models
Two Neural Network models were constructed, each with a distinct approach. The first, model_NN_1, utilized the original feature set with Word2Vec vectors, while the second, model_NN_2, employed the normalized feature set. Early stopping was implemented during training, and both models were evaluated for test accuracy, probing the predictive capabilities of the dataset using different feature sets.

This multi-faceted methodology aimed to discern the most accurate machine learning algorithm for the project while exploring the impact of outlier treatment strategies on the final results. The iterative optimization process involved fine-tuning hyperparameters, evaluating performance metrics, and considering the distinctive advantages offered by each algorithm. The inclusion of clustering and neural networks provided a comprehensive understanding of the dataset's structure and predictive potential.

