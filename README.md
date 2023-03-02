# Development Jobs & Skills Matching Web Application 


The main objective of this project is to demonstrate the entire process of an end to end real project. The final output is an effective prediction of matching jobs given a skill set.We are going to develop a data-driven solution for students to answer some questions : Which skills do I need to learn for my future IT job ?, They mostly want to understand the relationships between the jobs and the technologies.


## Problem Statement:

The project involves a client who is an IT education institute, and they have approached us with a problem. The IT industry is constantly evolving, which makes it difficult for their students to understand which skills they need to acquire for a particular job. The students have several questions, such as whether they need to learn C++ to become a data scientist or if they can use JavaScript for data analytics. The client has requested us to develop a data-driven solution that will help their students understand the relationship between different jobs and technologies.

## Business Case and KPIs:
The solution will have  a positive financial impact on these following:

**1)** Higher enrollment rate due to the higher certainty of students about their courses and learning strategies

**2)** Decrease in drop-out rate

**3)** Time saved for the academic advisors

**4)** Give answers for personlaized questions and different use cases




## Data Source

We have chosen this data source [Stack Overflow Developers Survey](https://insights.stackoverflow.com/survey).This dataset has a global distribution and a high volume. It's inclusive, updated and well structured. 

##The notebooks

* 01_preprocess:This code preprocesses a survey dataset by replacing certain string values with numerical values and splitting answers that contain multiple selections. Specifically, the code reads a CSV file from a path specified in DATA_PATH, performs some preprocessing steps on the data, and saves the resulting Pandas DataFrame as a pickle file at a path specified in EXPORT_PATH.

The preprocessing steps involve replacing certain string values with numerical values and splitting answers that contain multiple selections. Specifically, the REPLACE_DICT dictionary contains column names as keys, and sub-dictionaries as values, where the sub-dictionaries contain string values to be replaced with numerical values. The split_answers function is defined in an external script and takes a Pandas Series as input, splits each element on a semicolon delimiter, and returns a list of the resulting substrings.

Once the DataFrame has been preprocessed, it is saved as a pickle file using the to_pickle method. Overall, the purpose of this code seems to be to preprocess survey data in preparation for further analysis.

* 02_clean_data: This code cleans and filters a preprocessed survey dataset by one-hot encoding categorical variables, setting exclusion criteria, and dropping rows that meet the exclusion criteria. Specifically, the code reads a preprocessed DataFrame from a pickle file specified in DATA_PATH, one-hot encodes the columns specified in ROLE_COLS and TECH_COLS, and saves the resulting DataFrame as ohe_df.

The code then defines exclusion criteria to filter out certain rows from the DataFrame. The EXCLUDE_ROLES list contains role names to exclude from the dataset. The code calculates the frequency of roles and skills using the one-hot encoded columns, sets exclusion ranges for the frequency of roles and skills using N_ROLES_RANGE and N_TECH_RANGE, and excludes rows that fall outside of these ranges. Additionally, the code excludes rows that do not have an employment status of "Employed full-time" or "Employed part-time".

The code then merges the exclusion masks and saves the resulting DataFrame as exclude_df. The code calculates insights on the exclusion masks, such as the percentage of rows excluded by each mask and the percentage of rows excluded by multiple masks. Finally, the code applies the final exclusion mask to the preprocessed DataFrame to drop rows that meet the exclusion criteria and saves the resulting cleaned DataFrame as a pickle file specified in EXPORT_PATH.

Overall, the purpose of this code is to clean and filter a preprocessed survey dataset in preparation for further analysis, while setting specific exclusion criteria to ensure that the resulting dataset only includes rows that are relevant to the analysis.

* 03_visualize: This script seems to be a clustering analysis of skills that software developers have. The script reads a preprocessed dataset of survey responses and performs some data wrangling to filter out uninteresting jobs and one-hot-encode the skills (programming languages, databases, etc.) that respondents have used. Then, the script creates visualizations of the frequency of job roles and skills across the entire dataset, and also visualizes the frequency of skills within each job role. Next, the script performs hierarchical clustering on the skills using the t-SNE algorithm for dimensionality reduction, and uses the silhouette score to determine the optimal number of clusters. Finally, the script outputs the clusters of skills and the job roles that are associated with each cluster.

* 04_ensemble_model : The code provided trains and evaluates a stacked classifier model to predict developers' roles based on their technology experience. It uses a cleaned data set with one-hot-encoded columns for the technology and role columns, and it excludes some roles that have low representation.

The model consists of two base classifiers (a random forest and an elastic net logistic regression) and a logistic regression as a final estimator. It stacks them to improve the model's performance.

The get_train_test_data function is used to split the data into training and testing sets for each job. Then, the models dictionary is updated by training the stacked classifier on the training data for each job.

Finally, the model's performance is evaluated using the classification_report function from scikit-learn. The results are stored in two dictionaries, train_evaluation and test_evaluation, which contain precision, recall, f1-score, and support for each job.

The calculate_quality function is used to calculate the quality scores for the predictions made by the model. It takes as input the ground truth labels and predicted labels and a metric function (e.g., accuracy_score, f1_score) to compute the quality score for each job. The sort_values parameter determines whether to sort the scores by ascending order.

05_predict : This code loads a dictionary of pre-trained models from a pickled file and uses them to predict the probability that a respondent from a survey belongs to a specific job role, given the respondent's reported skills. The model is based on one-hot-encoded features of the survey respondents' skills and job roles.

The code also defines some variables and constants such as MODEL_DIR, MODEL_FILE, ROLE_COLS, TECH_COLS, and EXCLUDE_ROLES which are not used in this specific piece of code, but might be used in other parts of the code.

Lastly, the code specifies a list of skills (skills = ['Scala', 'Julia']), checks if these skills are part of the features used in the model, and generates a horizontal bar plot using Plotly to display the predicted probabilities for each job role.

## Scripts:

* SkillsJobModel.py: The code defines a class called SkillsJobModel that is used to load and make predictions using an ensemble of machine learning models trained to predict the probability of someone having a specific job title based on their skills. The class loads the models from a file, validates that all models have the same set of features, and then provides methods to predict the probability of having a specific job title based on a list of skills. The class uses one-hot encoding to represent the skills as features and then passes the encoded skills to each model to get a probability prediction. The class also includes validation steps to ensure that all user-provided skills are a part of the models' feature set.

* preprocessing.py : The code defines two functions for preprocessing data:

split_answers(data_series: pd.Series, delimiter=";") -> pd.Series: This function takes in a Pandas Series containing answers with a specified delimiter and splits each answer into a list of single answers. If the original Series does not contain multiple answers, the function returns the original Series. The function then returns a modified Series with each answer split into a list.

one_hot_encode(df: pd.DataFrame, columns): This function takes in a Pandas DataFrame and a list of column names. It one-hot-encodes the specified columns of the DataFrame and returns a new DataFrame with the one-hot-encoded columns concatenated. The function uses the MultiLabelBinarizer class from scikit-learn to perform the one-hot encoding. If the columns argument is not a list, the function raises a ValueError.

## Web Application: 

This is a Python script that defines a Dash app for a skills matching tool. The app allows users to select skills they have, and then shows them a bar chart indicating the probability of matching with different job roles. The app loads a saved machine learning model for making predictions, and uses the Plotly library to create the bar chart. The app also includes some CSS styling using Bootstrap. The script defines various functions to handle user inputs and generate the chart, and then sets up the layout and callbacks for the app.

This is how the user interface of our web applciation looks like, we selected specific skills related to Dta Sience, and fortunately, we obtained the expected result:
(user1)[https://raw.githubusercontent.com/MariemAmmar/Web_App_Matching_Skills_and_Jobs/main/Web_APP_Matching_Skills_and_Jobs/images/img%201.PNG]
(user2)[https://raw.githubusercontent.com/MariemAmmar/Web_App_Matching_Skills_and_Jobs/main/Web_APP_Matching_Skills_and_Jobs/images/img2.PNG]
(user3)[https://raw.githubusercontent.com/MariemAmmar/Web_App_Matching_Skills_and_Jobs/main/Web_APP_Matching_Skills_and_Jobs/images/img3.PNG]
