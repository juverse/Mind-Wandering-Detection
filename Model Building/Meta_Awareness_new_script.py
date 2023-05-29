#### Mulitclass Classification: Meta-Awareness: can we decide between MW aware and MW unaware ? ####
print("Mulitclass Classification: Meta-Awarenes")
## Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import modules for feature engineering and modelling
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
# pipeline
from imblearn.pipeline import Pipeline
# models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
# cross validation and hyperparameter tuning
from sklearn.model_selection import PredefinedSplit,  cross_validate, StratifiedKFold, GridSearchCV, cross_val_score
# balancing
from imblearn.over_sampling import RandomOverSampler, SMOTE
#accuracy
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer, classification_report, average_precision_score
#feature selection
from sklearn.feature_selection import SelectKBest, chi2
from statistics import mean
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import pickle

##### data
read_path = r"W:\WCT\04_Mind-Wandering-Labstudy\04_Daten\04_Prepared_data\00_Julia\Scripts\features_with_labels.csv"
save_path = r"W:\WCT\04_Mind-Wandering-Labstudy\04_Daten\04_Prepared_data\00_Julia\Scripts"
df = pd.read_csv(read_path)

labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
df["Awareness_all_new"] = labelencoder.fit_transform(df["Awareness_all"])
print(df["Awareness_all_new"].value_counts())

# 1  = task-related  631, 0  = aware  249, 2 = unaware   147
def get_X_y(train):
    FEATURES = [
        'Fixation Duration Mean [ms]', 'Fixation Duration Max [ms]', 'Fixation Duration Min [ms]', 'Fixation Duration Median [ms]', 'Fixation Duration Std [ms]', 'Fixation Duration Skew [ms]', 'Fixation Duration Quantil 25 [ms]', 'Fixation Duration Quantil 75 [ms]',
        'Saccade Duration Mean [ms]', 'Saccade Duration Max [ms]', 'Saccade Duration Min [ms]', 'Saccade Duration Median [ms]', 'Saccade Duration Std [ms]', 'Saccade Duration Skew [ms]', 'Saccade Duration Quantil 25 [ms]', 'Saccade Duration Quantil 75 [ms]', 
        'Blink Duration Mean [ms]', 'Blink Duration Max [ms]', 'Blink Duration Min [ms]', 'Blink Duration Median [ms]', 'Blink Duration Std [ms]', 'Blink Duration Skew [ms]', 'Blink Duration Quantil 25 [ms]', 'Blink Duration Quantil 75 [ms]', 'Fixation Duration Kurtosis [ms]',
        'Saccade Duration Kurtosis [ms]',
        'Blink Duration Kurtosis [ms]', 
        'Fixation Saccade Ratio Mean', 'Fixation Saccade Ratio Max', 'Fixation Saccade Ratio Min', 'Fixation Saccade Ratio Median', 'Fixation Saccade Ratio Std', 'Fixation Saccade Ratio Skew', 'Fixation Saccade Ratio Kurtosis', 
        'Fixation Number', 'Blink Number', 
        'Fixation Dispersion X Mean [px]', 'Fixation Dispersion X Max [px]', 'Fixation Dispersion X Min [px]', 'Fixation Dispersion X Median [px]', 'Fixation Dispersion X Std [px]', 'Fixation Dispersion X Skew [px]', 'Fixation Dispersion X Quantil 25 [px]', 'Fixation Dispersion X Quantil 75 [px]', 
        'Fixation Dispersion Y Mean [px]', 'Fixation Dispersion Y Max [px]', 'Fixation Dispersion Y Min [px]', 'Fixation Dispersion Y Median [px]', 'Fixation Dispersion Y Std [px]', 'Fixation Dispersion Y Skew [px]', 'Fixation Dispersion Y Quantil 25 [px]', 'Fixation Dispersion Y Quantil 75 [px]', 'Fixation Dispersion X Kurtosis [px]', 'Fixation Dispersion Y Kurtosis [px]', 
        'Saccade Amplitude Mean [°]', 'Saccade Amplitude Max [°]', 'Saccade Amplitude Min [°]', 'Saccade Amplitude Median [°]', 'Saccade Amplitude Std [°]', 'Saccade Amplitude Skew [°]', 'Saccade Amplitude Quantil 25 [°]', 'Saccade Amplitude Quantil 75 [°]', 'Saccade Amplitude Kurtosis [°]',
        'Saccade Acceleration Average [°/s²] Mean', 'Saccade Acceleration Average [°/s²] Max', 'Saccade Acceleration Average [°/s²] Min', 'Saccade Acceleration Average [°/s²] Median', 'Saccade Acceleration Average [°/s²] Std', 'Saccade Acceleration Average [°/s²] Skew]', 'Saccade Acceleration Average [°/s²] Quantil 25]', 'Saccade Acceleration Average [°/s²] Quantil 75]',
        'Saccade Acceleration Peak [°/s²] Mean', 'Saccade Acceleration Peak [°/s²] Max', 'Saccade Acceleration Peak [°/s²] Min', 'Saccade Acceleration Peak [°/s²] Median', 'Saccade Acceleration Peak [°/s²] Std', 'Saccade Acceleration Peak [°/s²] Skew]', 'Saccade Acceleration Peak [°/s²] Quantil 25]', 'Saccade Acceleration Peak [°/s²] Quantil 75]', 'Saccade Deceleration Peak [°/s²] Mean', 
        'Saccade Deceleration Peak [°/s²] Max', 'Saccade Deceleration Peak [°/s²] Min', 'Saccade Deceleration Peak [°/s²] Median', 'Saccade Deceleration Peak [°/s²] Std', 'Saccade Deceleration Peak [°/s²] Skew]', 'Saccade Deceleration Peak [°/s²] Quantil 25]', 'Saccade Deceleration Peak [°/s²] Quantil 75]', 
        'Saccade Velocity Average [°/s²] Mean', 'Saccade Velocity Average [°/s²] Max', 'Saccade Velocity Average [°/s²] Min', 'Saccade Velocity Average [°/s²] Median', 'Saccade Velocity Average [°/s²] Std', 'Saccade Velocity Average [°/s²] Skew]', 'Saccade Velocity Average [°/s²] Quantil 25]', 'Saccade Velocity Average [°/s²] Quantil 75]', 
        'Saccade Velocity Peak [°/s²] Mean', 'Saccade Velocity Peak [°/s²] Max', 'Saccade Velocity Peak [°/s²] Min', 'Saccade Velocity Peak [°/s²] Median', 'Saccade Velocity Peak [°/s²] Std', 'Saccade Velocity Peak [°/s²] Skew]', 'Saccade Velocity Peak [°/s²] Quantil 25]', 'Saccade Velocity Peak [°/s²] Quantil 75]', 
        'Saccade Velocity Peak [%] Mean', 'Saccade Velocity Peak [%] Max', 'Saccade Velocity Peak [%] Min', 'Saccade Velocity Peak [%] Median', 'Saccade Velocity Peak [%] Std', 'Saccade Velocity Peak [%] Skew]', 'Saccade Velocity Peak [%] Quantil 25]', 'Saccade Velocity Peak [%] Quantil 75]', 
        'Saccade Acceleration Average [°/s²] Kurtosis', 'Saccade Acceleration Peak [°/s²] Kurtosis', 'Saccade Deceleration Peak [°/s²] Kurtosis', 'Saccade Velocity Average [°/s²] Kurtosis', 'Saccade Velocity Peak [°/s²] Kurtosis', 'Saccade Velocity Peak [%] Kurtosis', 
        'Saccade Length Mean [px]', 'Saccade Length Max [px]', 'Saccade Length Min [px]', 'Saccade Length Median [px]', 'Saccade Length Std [px]', 'Saccade Length Skew [px]]', 'Saccade Length Quantil 25 [px]]', 'Saccade Length Quantil 75 [px]]', 'Saccade Length Kurtosis [px]', 
        'Fixation Average Pupil Diameter [mm] Mean', 'Fixation Average Pupil Diameter [mm] Max', 'Fixation Average Pupil Diameter [mm] Min', 'Fixation Average Pupil Diameter [mm] Median', 'Fixation Average Pupil Diameter [mm] Std', 'Fixation Average Pupil Diameter [mm] Skew', 'Fixation Average Pupil Diameter [mm] Quantil25', 'Fixation Average Pupil Diameter [mm] Quantil75',
        'Fixation Average Pupil Diameter [mm] Kurtosis', 
        'Veregence Angles Mean [rad]', 'Veregence Angles Std [rad]', 
        'Pupil Distance Mean [px]', 'Pupil Distance Std [px]'
    ]

    GROUPS = "Participant"

    TARGET = "Awareness_all_new"

    X = train[FEATURES]
    y = train[TARGET]
    groups = train[GROUPS]
    return X, y, groups

X, y, groups = get_X_y(df)

#testing
#X = X.head(100)
#y = y.head(100)
# Transform y to one-hot encoding format
lb = LabelBinarizer()
y_onehot = lb.fit_transform(y)

### Classification models with Pipeline

# SVM Pipeline
imputer =  SimpleImputer(fill_value='missing')
scaler = StandardScaler()

svm = SVC()
steps = [('imputer', imputer), ('scaler',scaler), ('over', None), ('svm', svm)]
pipe_svm = Pipeline(steps=steps)

param_grid_svm = {
            #balancing method
            'over': [SMOTE(random_state= 4),RandomOverSampler(random_state=4), None],
            # balanced = smaller classes get more weights
            'svm__class_weight': ['balanced', None],
            # C parameter adds a penalty for each misclassified data point. 
            'svm__C': [0.1, 1, 5, 10, 100], 
            #When the value of gamma is very small, the model is too constrained and cannot capture the complexity or “shape” of the data.
            'svm__gamma':  ["scale", "auto", 1, 0.1, 0.01, 0.001, 0.0001],
            #tells SVM how close a data point is to the hyperplane, a data point close to the boundary means a low-confidence decision
            'svm__decision_function_shape': ['ovr', 'ovo'],
            # kernel function, to go to a higher dimension
            'svm__kernel': ["linear", "rbf"]
            } 

num_classes = 3
inner_cv = StratifiedKFold(n_splits=num_classes, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=num_classes, shuffle=True, random_state=42)

# Define the list of scoring metrics to compute
# Define scorer functions to calculate f1 score for each class separately
recall_per_class = {'recall_score_class{}'.format(i): make_scorer(recall_score, average=None, labels=[i]) for i in range(num_classes)}
f1_per_class = {'f1_score_class{}'.format(i): make_scorer(f1_score, average=None, labels=[i]) for i in range(num_classes)}
percision_per_class = {'precision_score_class{}'.format(i): make_scorer(precision_score, average=None, labels=[i]) for i in range(num_classes)}

# define Macro scores- how good are the prediction for all three classes 
scoring = {
    'f1_macro': make_scorer(f1_score, average='macro'),
    'recall_macro':make_scorer(recall_score, average='macro'),
    'precision_macro':  make_scorer(precision_score, average='macro'),
}
# merge the dictonaries
scoring.update(recall_per_class)
scoring.update(f1_per_class)
scoring.update(percision_per_class)

# Nested CV with parameter optimization
# The inner split is used to tune the hyperparameters of the model, and the outer split is used to estimate the performance of the model on new, unseen data

# The inner loop of cross-validation is used to tune the hyperparameters of the model on the training set. 
# This is done by splitting the training set into a further set of training and validation sets, 
# and evaluating the model's performance on the validation set for each combination of hyperparameters. This allows you to select the best hyperparameters for the model.
clf = GridSearchCV(estimator=pipe_svm,
                    param_grid=param_grid_svm, 
                    cv=inner_cv,
                    scoring = 'f1_macro',
                    refit = True
                    )

#The outer loop of cross-validation is used to estimate the performance of the model on new, unseen data.
nested_score = cross_validate(clf, 
                    X=X, 
                    y= y, 
                    cv=outer_cv,
                    scoring=scoring,
                    return_estimator=True
                    )


# Get the best estimator from the GridSearchCV object for each outer fold
estimator = nested_score['estimator'][0]
# extract best estimator
best_estimator = estimator.best_estimator_
#
#Get the estimator object from the cross-validation results
estimator = nested_score['estimator'][0]
## extract best estimator
best_estimator = estimator.best_estimator_
##extract best parameters
best_params_svm = best_estimator.named_steps['svm'].get_params()
## extract the crossvalidation results from the estsimator object
estimator_results = estimator.cv_results_

## Convert cv_results_ to DataFrame and print
cv_results_df = pd.DataFrame.from_dict(estimator_results)
cv_results_df.to_csv(save_path + '\\' + 'svm_cv_results_ma.csv', index = False)

# Extract rows with over = RandomOverSampler and find row with highest mean_test_score
random_sampler_rows= cv_results_df.loc[cv_results_df['param_over'].astype('string') == 'RandomOverSampler(random_state=4)']
# Extract row with highest mean_test_score for RandomOverSampler
max_random_sampler = random_sampler_rows.loc[random_sampler_rows['mean_test_score'].idxmax(), ['params', 'mean_test_score', 'std_test_score']]

# Extract rows with over = SMOTE(random_state=4) and find row with highest mean_test_score
smote_rows= cv_results_df.loc[cv_results_df['param_over'].astype('string') == 'SMOTE(random_state=4)']
# Extract row with highest mean_test_score for SMOTE(random_state=4)
max_smote = smote_rows.loc[smote_rows['mean_test_score'].idxmax(), ['params', 'mean_test_score', 'std_test_score']]

#print the results for SVM 
print("######")
print("SVM")
print('best random sampler:', max_random_sampler)
print('best smote:', max_smote, flush = True)
print(" best estimator: ")
print(best_estimator)
print("best paramter")
print(best_params_svm)

# Print the overall f1 score, per-class f1 scores, and average precision scores
print('f1')
print("Class 0 F1 score: ", mean(nested_score['test_f1_score_class0']) )
print("Class 1 F1 score: ", mean(nested_score['test_f1_score_class1']) )
print("Class 2 F1 score: ", mean(nested_score['test_f1_score_class2']) )
print('recall')
print("Class 0 recall score: ", mean(nested_score['test_recall_score_class0']) )
print("Class 1 recall score: ", mean(nested_score['test_recall_score_class1']) )
print("Class 2 recall score: ", mean(nested_score['test_recall_score_class2']) )
print('precision')
print("Class 0 precision score: ", mean(nested_score['test_precision_score_class0']) )
print("Class 1 precision score: ", mean(nested_score['test_precision_score_class1']) )
print("Class 2 precision score: ", mean(nested_score['test_precision_score_class2']) )
print("Macro F1 score: ", mean(nested_score['test_f1_macro']) )
print("Macro precision score: ", mean(nested_score['test_precision_macro']) )
print("Macro recall score: ", mean(nested_score['test_recall_macro']) )
print("######")


#RandomForrest Pipeline
imputer =  SimpleImputer(fill_value='missing')
scaler = StandardScaler()

random_forest = RandomForestClassifier()
steps = [('imputer', imputer), ('scaler',scaler), ('over', None), ('random_forest', random_forest)]
pipe_random_forrest = Pipeline(steps=steps)
# 
#defining parameter range
param_grid_random_forrest= {
            #balancing method
            'over': [SMOTE(random_state= 4), RandomOverSampler(random_state=4), None],
            'random_forest__class_weight':["balanced", None],
            # Method of selecting samples for training each tree
            'random_forest__bootstrap': [True, False],
            # Maximum number of levels in tree
            'random_forest__max_depth':[6, 10, 20, 30, 50, 80, None],
             #Number of features to consider at every split
            'random_forest__max_features': ['sqrt', None],
             # Minimum number of samples required to split a node
            'random_forest__min_samples_leaf':[1, 2, 4],
             #Minimum number of samples required at each leaf node
            'random_forest__min_samples_split': [2, 5, 10],
             #Number of trees in random forest
            'random_forest__n_estimators': [50,70,100, 200],
            }       

num_classes = 3
inner_cv = StratifiedKFold(n_splits=num_classes, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=num_classes, shuffle=True, random_state=42)

# Define the list of scoring metrics to compute
# Define scorer functions to calculate f1 score for each class separately
recall_per_class = {'recall_score_class{}'.format(i): make_scorer(recall_score, average=None, labels=[i]) for i in range(num_classes)}
f1_per_class = {'f1_score_class{}'.format(i): make_scorer(f1_score, average=None, labels=[i]) for i in range(num_classes)}
percision_per_class = {'precision_score_class{}'.format(i): make_scorer(precision_score, average=None, labels=[i]) for i in range(num_classes)}

# define Macro scores- how good are the prediction for all three classes 
scoring = {
    'f1_macro': make_scorer(f1_score, average='macro'),
    'recall_macro':make_scorer(recall_score, average='macro'),
    'precision_macro':  make_scorer(precision_score, average='macro'),
}
# merge the dictonaries
scoring.update(recall_per_class)
scoring.update(f1_per_class)
scoring.update(percision_per_class)

# Nested CV with parameter optimization
# The inner split is used to tune the hyperparameters of the model, and the outer split is used to estimate the performance of the model on new, unseen data

# The inner loop of cross-validation is used to tune the hyperparameters of the model on the training set. 
# This is done by splitting the training set into a further set of training and validation sets, 
# and evaluating the model's performance on the validation set for each combination of hyperparameters. This allows you to select the best hyperparameters for the model.
clf = GridSearchCV(estimator=pipe_random_forrest,
                    param_grid=param_grid_random_forrest, 
                    cv=inner_cv,
                    scoring = 'f1_macro',
                    refit = True
                    )

#The outer loop of cross-validation is used to estimate the performance of the model on new, unseen data.
nested_score = cross_validate(clf, 
                    X=X, 
                    y= y, 
                    cv=outer_cv,
                    scoring=scoring,
                    return_estimator=True
                    )


# Get the best estimator from the GridSearchCV object for each outer fold
estimator = nested_score['estimator'][0]
# extract best estimator
best_estimator = estimator.best_estimator_
#
#Get the estimator object from the cross-validation results
estimator = nested_score['estimator'][0]
## extract best estimator
best_estimator = estimator.best_estimator_
##extract best parameters
best_params_svm = best_estimator.named_steps['random_forest'].get_params()
## extract the crossvalidation results from the estsimator object
estimator_results = estimator.cv_results_

## Convert cv_results_ to DataFrame and print
cv_results_df = pd.DataFrame.from_dict(estimator_results)
cv_results_df.to_csv(save_path + '\\' + 'random_forest_cv_results_ma.csv', index = False)

# Extract rows with over = RandomOverSampler and find row with highest mean_test_score
random_sampler_rows= cv_results_df.loc[cv_results_df['param_over'].astype('string') == 'RandomOverSampler(random_state=4)']
# Extract row with highest mean_test_score for RandomOverSampler
max_random_sampler = random_sampler_rows.loc[random_sampler_rows['mean_test_score'].idxmax(), ['params', 'mean_test_score', 'std_test_score']]

# Extract rows with over = SMOTE(random_state=4) and find row with highest mean_test_score
smote_rows= cv_results_df.loc[cv_results_df['param_over'].astype('string') == 'SMOTE(random_state=4)']
# Extract row with highest mean_test_score for SMOTE(random_state=4)
max_smote = smote_rows.loc[smote_rows['mean_test_score'].idxmax(), ['params', 'mean_test_score', 'std_test_score']]

#print the results for random_forest 
print("######")
print("random_forest")
print('best random sampler:', max_random_sampler)
print('best smote:', max_smote, flush = True)
print("best estimator: ")
print(best_estimator)
print("best paramter")
print(best_params_svm)

# Print the overall f1 score, per-class f1 scores, and average precision scores
print('f1')
print("Class 0 F1 score: ", mean(nested_score['test_f1_score_class0']) )
print("Class 1 F1 score: ", mean(nested_score['test_f1_score_class1']) )
print("Class 2 F1 score: ", mean(nested_score['test_f1_score_class2']) )
print('recall')
print("Class 0 recall score: ", mean(nested_score['test_recall_score_class0']) )
print("Class 1 recall score: ", mean(nested_score['test_recall_score_class1']) )
print("Class 2 recall score: ", mean(nested_score['test_recall_score_class2']) )
print('precision')
print("Class 0 precision score: ", mean(nested_score['test_precision_score_class0']) )
print("Class 1 precision score: ", mean(nested_score['test_precision_score_class1']) )
print("Class 2 precision score: ", mean(nested_score['test_precision_score_class2']) )
print("Macro F1 score: ", mean(nested_score['test_f1_macro']) )
print("Macro precision score: ", mean(nested_score['test_precision_macro']) )
print("Macro recall score: ", mean(nested_score['test_recall_macro']) )
print("######")

## naive bayes Pipeline
imputer =  SimpleImputer(fill_value='missing')
scaler = StandardScaler()

naive_bayes =  GaussianNB()
steps = [('imputer', imputer), ('scaler',scaler), ('over', None), ('naive_bayes', naive_bayes)]
pipe_naive_bayes = Pipeline(steps=steps)
 # sample weights
    #sample_weights = class_weight.compute_sample_weight(class_weight = 'balanced',  y = y_train)
    ## # Fit Model on Train
    #pipe.fit(X_train, y_train, **{'model__sample_weight': sample_weights})

# defining parameter range
param_grid_naive_bayes= {
            #balancing method
            'over': [SMOTE(random_state= 4), RandomOverSampler(random_state=4), None],
            # The prior probabilities can be used to adjust the probability of a particular class based on the presence or absence of certain features
            #'naive_bayes__priors':[[0.24, 0.62,0.14], None],
            # The portion of the largest variance of all features that is added to variances for calculation stability.
            'naive_bayes__var_smoothing': np.logspace(0,-9, num=100)
            }       
                          
num_classes = 3
inner_cv = StratifiedKFold(n_splits=num_classes, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=num_classes, shuffle=True, random_state=42)

# Define the list of scoring metrics to compute
# Define scorer functions to calculate f1 score for each class separately
recall_per_class = {'recall_score_class{}'.format(i): make_scorer(recall_score, average=None, labels=[i]) for i in range(num_classes)}
f1_per_class = {'f1_score_class{}'.format(i): make_scorer(f1_score, average=None, labels=[i]) for i in range(num_classes)}
percision_per_class = {'precision_score_class{}'.format(i): make_scorer(precision_score, average=None, labels=[i]) for i in range(num_classes)}

# define Macro scores- how good are the prediction for all three classes 
scoring = {
    'f1_macro': make_scorer(f1_score, average='macro'),
    'recall_macro':make_scorer(recall_score, average='macro'),
    'precision_macro':  make_scorer(precision_score, average='macro'),
}
# merge the dictonaries
scoring.update(recall_per_class)
scoring.update(f1_per_class)
scoring.update(percision_per_class)

# Nested CV with parameter optimization
# The inner split is used to tune the hyperparameters of the model, and the outer split is used to estimate the performance of the model on new, unseen data

# The inner loop of cross-validation is used to tune the hyperparameters of the model on the training set. 
# This is done by splitting the training set into a further set of training and validation sets, 
# and evaluating the model's performance on the validation set for each combination of hyperparameters. This allows you to select the best hyperparameters for the model.
clf = GridSearchCV(estimator=pipe_naive_bayes,
                    param_grid=param_grid_naive_bayes, 
                    cv=inner_cv,
                    scoring = 'f1_macro',
                    refit = True
                    )

#The outer loop of cross-validation is used to estimate the performance of the model on new, unseen data.
nested_score = cross_validate(clf, 
                    X=X, 
                    y= y, 
                    cv=outer_cv,
                    scoring=scoring,
                    return_estimator=True
                    )


# Get the best estimator from the GridSearchCV object for each outer fold
estimator = nested_score['estimator'][0]
# extract best estimator
best_estimator = estimator.best_estimator_
#
#Get the estimator object from the cross-validation results
estimator = nested_score['estimator'][0]
## extract best estimator
best_estimator = estimator.best_estimator_
##extract best parameters
best_params_svm = best_estimator.named_steps['naive_bayes'].get_params()
## extract the crossvalidation results from the estsimator object
estimator_results = estimator.cv_results_

## Convert cv_results_ to DataFrame and print
cv_results_df = pd.DataFrame.from_dict(estimator_results)
cv_results_df.to_csv(save_path + '\\' + 'naive_bayes_cv_results_ma.csv', index = False)

# Extract rows with over = RandomOverSampler and find row with highest mean_test_score
random_sampler_rows= cv_results_df.loc[cv_results_df['param_over'].astype('string') == 'RandomOverSampler(random_state=4)']
# Extract row with highest mean_test_score for RandomOverSampler
max_random_sampler = random_sampler_rows.loc[random_sampler_rows['mean_test_score'].idxmax(), ['params', 'mean_test_score', 'std_test_score']]

# Extract rows with over = SMOTE(random_state=4) and find row with highest mean_test_score
smote_rows= cv_results_df.loc[cv_results_df['param_over'].astype('string') == 'SMOTE(random_state=4)']
# Extract row with highest mean_test_score for SMOTE(random_state=4)
max_smote = smote_rows.loc[smote_rows['mean_test_score'].idxmax(), ['params', 'mean_test_score', 'std_test_score']]

#print the results for naive_bayes
print("######")
print("naive_bayes")
print('best random sampler:', max_random_sampler)
print('best smote:', max_smote, flush = True)
print(" best estimator: ")
print(best_estimator)
print("best paramter")
print(best_params_svm)

# Print the overall f1 score, per-class f1 scores, and average precision scores
print('f1')
print("Class 0 F1 score: ", mean(nested_score['test_f1_score_class0']) )
print("Class 1 F1 score: ", mean(nested_score['test_f1_score_class1']) )
print("Class 2 F1 score: ", mean(nested_score['test_f1_score_class2']) )
print('recall')
print("Class 0 recall score: ", mean(nested_score['test_recall_score_class0']) )
print("Class 1 recall score: ", mean(nested_score['test_recall_score_class1']) )
print("Class 2 recall score: ", mean(nested_score['test_recall_score_class2']) )
print('precision')
print("Class 0 precision score: ", mean(nested_score['test_precision_score_class0']) )
print("Class 1 precision score: ", mean(nested_score['test_precision_score_class1']) )
print("Class 2 precision score: ", mean(nested_score['test_precision_score_class2']) )
print("Macro F1 score: ", mean(nested_score['test_f1_macro']) )
print("Macro precision score: ", mean(nested_score['test_precision_macro']) )
print("Macro recall score: ", mean(nested_score['test_recall_macro']) )
print("######")

#XGBoost Pipeline
imputer =  SimpleImputer(fill_value='missing')
scaler = StandardScaler()

#compute class weights 
#class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
## class1 is the biggest, we want to give less weight to this one
#class_0_weight = class_weights[1] / class_weights[0]
#class_2_weight = class_weights[1] / class_weights[2]

xgboost =   xgb.XGBClassifier()
steps = [('imputer', imputer), ('scaler',scaler), ('over', None), ('xgboost', xgboost)]
pipe_xgboost = Pipeline(steps=steps)


# defining parameter range
param_grid_xgboost= {
            #balancing method
            'over': [SMOTE(random_state= 4), RandomOverSampler(random_state=4)],
            # Subsample ratio of the training instances, or the proportion of samples used to train each tree
            "xgboost__subsample": np.arange(0.6,1,0.05),
            #Maximum depth of each decision tree in the ensemble
            "xgboost__max_depth": np.arange(3,10,1),
            #Number of trees in the ensemble
            "xgboost__n_estimators": [1000, 700, 500],
            #Subsample ratio of columns when constructing each tree. 
            "xgboost__colsample_bytree": [0.1,0.5, 1.0],
           #not possible in muliclass calssification problem?
            #'xgboost__scale_pos_weight': class_weights #[class_0_weight, class_2_weight, None]
            }       
                          
num_classes = 3
inner_cv = StratifiedKFold(n_splits=num_classes, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=num_classes, shuffle=True, random_state=42)

# Define the list of scoring metrics to compute
# Define scorer functions to calculate f1 score for each class separately
recall_per_class = {'recall_score_class{}'.format(i): make_scorer(recall_score, average=None, labels=[i]) for i in range(num_classes)}
f1_per_class = {'f1_score_class{}'.format(i): make_scorer(f1_score, average=None, labels=[i]) for i in range(num_classes)}
percision_per_class = {'precision_score_class{}'.format(i): make_scorer(precision_score, average=None, labels=[i]) for i in range(num_classes)}

# define Macro scores- how good are the prediction for all three classes 
scoring = {
    'f1_macro': make_scorer(f1_score, average='macro'),
    'recall_macro':make_scorer(recall_score, average='macro'),
    'precision_macro':  make_scorer(precision_score, average='macro'),
}
# merge the dictonaries
scoring.update(recall_per_class)
scoring.update(f1_per_class)
scoring.update(percision_per_class)

# Nested CV with parameter optimization
# The inner split is used to tune the hyperparameters of the model, and the outer split is used to estimate the performance of the model on new, unseen data

# The inner loop of cross-validation is used to tune the hyperparameters of the model on the training set. 
# This is done by splitting the training set into a further set of training and validation sets, 
# and evaluating the model's performance on the validation set for each combination of hyperparameters. This allows you to select the best hyperparameters for the model.
clf = GridSearchCV(estimator=pipe_xgboost,
                    param_grid=param_grid_xgboost, 
                    cv=inner_cv,
                    scoring = 'f1_macro',
                    refit = True
                    )

#The outer loop of cross-validation is used to estimate the performance of the model on new, unseen data.
nested_score = cross_validate(clf, 
                    X=X, 
                    y= y, 
                    cv=outer_cv,
                    scoring=scoring,
                    return_estimator=True
                    )


# Get the best estimator from the GridSearchCV object for each outer fold
estimator = nested_score['estimator'][0]
# extract best estimator
best_estimator = estimator.best_estimator_
#Get the estimator object from the cross-validation results
estimator = nested_score['estimator'][0]
## extract best estimator
best_estimator = estimator.best_estimator_
##extract best parameters
best_params_svm = best_estimator.named_steps['xgboost'].get_params()
#
feature_importances = best_estimator.named_steps['xgboost'].feature_importances_
print(feature_importances)
## extract the crossvalidation results from the estsimator object
estimator_results = estimator.cv_results_

## Convert cv_results_ to DataFrame and print
cv_results_df = pd.DataFrame.from_dict(estimator_results)
cv_results_df.to_csv(save_path + '\\' + 'xgboost_cv_results_ma.csv', index = False)

# Extract rows with over = RandomOverSampler and find row with highest mean_test_score
random_sampler_rows= cv_results_df.loc[cv_results_df['param_over'].astype('string') == 'RandomOverSampler(random_state=4)']
# Extract row with highest mean_test_score for RandomOverSampler
max_random_sampler = random_sampler_rows.loc[random_sampler_rows['mean_test_score'].idxmax(), ['params', 'mean_test_score', 'std_test_score']]

# Extract rows with over = SMOTE(random_state=4) and find row with highest mean_test_score
smote_rows= cv_results_df.loc[cv_results_df['param_over'].astype('string') == 'SMOTE(random_state=4)']
# Extract row with highest mean_test_score for SMOTE(random_state=4)
max_smote = smote_rows.loc[smote_rows['mean_test_score'].idxmax(), ['params', 'mean_test_score', 'std_test_score']]

#print the results for xgboost
print("######")
print("xgboost")
print('best random sampler:', max_random_sampler)
print('best smote:', max_smote, flush = True)
print("best estimator: ")
print(best_estimator)
print("best paramter")
print(best_params_svm)

# Print the overall f1 score, per-class f1 scores, and average precision scores
print('f1')
print("Class 0 F1 score: ", mean(nested_score['test_f1_score_class0']) )
print("Class 1 F1 score: ", mean(nested_score['test_f1_score_class1']) )
print("Class 2 F1 score: ", mean(nested_score['test_f1_score_class2']) )
print('recall')
print("Class 0 recall score: ", mean(nested_score['test_recall_score_class0']) )
print("Class 1 recall score: ", mean(nested_score['test_recall_score_class1']) )
print("Class 2 recall score: ", mean(nested_score['test_recall_score_class2']) )
print('precision')
print("Class 0 precision score: ", mean(nested_score['test_precision_score_class0']) )
print("Class 1 precision score: ", mean(nested_score['test_precision_score_class1']) )
print("Class 2 precision score: ", mean(nested_score['test_precision_score_class2']) )
print("Macro F1 score: ", mean(nested_score['test_f1_macro']) )
print("Macro precision score: ", mean(nested_score['test_precision_macro']) )
print("Macro recall score: ", mean(nested_score['test_recall_macro']) )
print("######")


# MLP Pipeline
imputer =  SimpleImputer(fill_value='missing')
scaler = StandardScaler()

mlp =   MLPClassifier()
steps = [('imputer', imputer), ('scaler',scaler), ('over', None), ('mlp', mlp)]
pipe_mlp = Pipeline(steps=steps)

# defining parameter range
param_grid_mlp= {
            #balancing method
            'over': [SMOTE(random_state= 4), RandomOverSampler(random_state=4)],
            'mlp__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'mlp__activation': ['tanh', 'relu'],
            'mlp__solver': ['sgd', 'adam', 'lbfgs'],
            'mlp__alpha': [0.06, 0.05, 0.004],
            'mlp__learning_rate': ['constant','adaptive'],
            }       

num_classes = 3
inner_cv = StratifiedKFold(n_splits=num_classes, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=num_classes, shuffle=True, random_state=42)

# Define the list of scoring metrics to compute
# Define scorer functions to calculate f1 score for each class separately
recall_per_class = {'recall_score_class{}'.format(i): make_scorer(recall_score, average=None, labels=[i]) for i in range(num_classes)}
f1_per_class = {'f1_score_class{}'.format(i): make_scorer(f1_score, average=None, labels=[i]) for i in range(num_classes)}
percision_per_class = {'precision_score_class{}'.format(i): make_scorer(precision_score, average=None, labels=[i]) for i in range(num_classes)}

# define Macro scores- how good are the prediction for all three classes 
scoring = {
    'f1_macro': make_scorer(f1_score, average='macro'),
    'recall_macro':make_scorer(recall_score, average='macro'),
    'precision_macro':  make_scorer(precision_score, average='macro'),
}
# merge the dictonaries
scoring.update(recall_per_class)
scoring.update(f1_per_class)
scoring.update(percision_per_class)

# Nested CV with parameter optimization
# The inner split is used to tune the hyperparameters of the model, and the outer split is used to estimate the performance of the model on new, unseen data

# The inner loop of cross-validation is used to tune the hyperparameters of the model on the training set. 
# This is done by splitting the training set into a further set of training and validation sets, 
# and evaluating the model's performance on the validation set for each combination of hyperparameters. This allows you to select the best hyperparameters for the model.
clf = GridSearchCV(estimator=pipe_mlp,
                    param_grid=param_grid_mlp, 
                    cv=inner_cv,
                    scoring = 'f1_macro',
                    refit = True
                    )

#The outer loop of cross-validation is used to estimate the performance of the model on new, unseen data.
nested_score = cross_validate(clf, 
                    X=X, 
                    y= y, 
                    cv=outer_cv,
                    scoring=scoring,
                    return_estimator=True
                    )


# Get the best estimator from the GridSearchCV object for each outer fold
estimator = nested_score['estimator'][0]
# extract best estimator
best_estimator = estimator.best_estimator_
#Get the estimator object from the cross-validation results
estimator = nested_score['estimator'][0]
## extract best estimator
best_estimator = estimator.best_estimator_
##extract best parameters
best_params_svm = best_estimator.named_steps['mlp'].get_params()
## extract the crossvalidation results from the estsimator object
estimator_results = estimator.cv_results_

## Convert cv_results_ to DataFrame and print
cv_results_df = pd.DataFrame.from_dict(estimator_results)
cv_results_df.to_csv(save_path + '\\' + 'mlp_cv_results_ma.csv', index = False)

# Extract rows with over = RandomOverSampler and find row with highest mean_test_score
random_sampler_rows= cv_results_df.loc[cv_results_df['param_over'].astype('string') == 'RandomOverSampler(random_state=4)']
# Extract row with highest mean_test_score for RandomOverSampler
max_random_sampler = random_sampler_rows.loc[random_sampler_rows['mean_test_score'].idxmax(), ['params', 'mean_test_score', 'std_test_score']]

# Extract rows with over = SMOTE(random_state=4) and find row with highest mean_test_score
smote_rows= cv_results_df.loc[cv_results_df['param_over'].astype('string') == 'SMOTE(random_state=4)']
# Extract row with highest mean_test_score for SMOTE(random_state=4)
max_smote = smote_rows.loc[smote_rows['mean_test_score'].idxmax(), ['params', 'mean_test_score', 'std_test_score']]

#print the results for mlp
print("######")
print("mlp")
print('best random sampler:', max_random_sampler)
print('best smote:', max_smote, flush = True)
print(" best estimator: ")
print(best_estimator)
print("best paramter")
print(best_params_svm)

# Print the overall f1 score, per-class f1 scores, and average precision scores
print('f1')
print("Class 0 F1 score: ", mean(nested_score['test_f1_score_class0']) )
print("Class 1 F1 score: ", mean(nested_score['test_f1_score_class1']) )
print("Class 2 F1 score: ", mean(nested_score['test_f1_score_class2']) )
print('recall')
print("Class 0 recall score: ", mean(nested_score['test_recall_score_class0']) )
print("Class 1 recall score: ", mean(nested_score['test_recall_score_class1']) )
print("Class 2 recall score: ", mean(nested_score['test_recall_score_class2']) )
print('precision')
print("Class 0 precision score: ", mean(nested_score['test_precision_score_class0']) )
print("Class 1 precision score: ", mean(nested_score['test_precision_score_class1']) )
print("Class 2 precision score: ", mean(nested_score['test_precision_score_class2']) )
print("Macro F1 score: ", mean(nested_score['test_f1_macro']) )
print("Macro precision score: ", mean(nested_score['test_precision_macro']) )
print("Macro recall score: ", mean(nested_score['test_recall_macro']) )
print("######")
