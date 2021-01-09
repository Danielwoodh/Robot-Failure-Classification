# Importing the libraries

# RegEx
import re

# Linear Algebra
import numpy as np

# Data-processing
import pandas as pd

# Graphing
import seaborn as sns
import matplotlib.pyplot as plt
from cycler import cycler

# Parameter tuning & scoring
from sklearn.model_selection import (
    RandomizedSearchCV, ParameterSampler, 
    train_test_split, ShuffleSplit
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report,
    confusion_matrix, plot_confusion_matrix,
    multilabel_confusion_matrix
)

# SMOTE
from imblearn.over_sampling import SMOTE


def process_data(filename):
    '''
    This function transforms the original dataset so that it's in a tabular format
    
    Inputs:
      filename | string | The path to the dataset

    Outputs:
      df | pandas.DataFrame | The dataframe containing the transformed dataset
    '''
    
    # Creating an empty string for the window
    windows = ''
    
    # Opening and reading the desired file
    with open(filename) as f:
        windows += f.read()
    
    # Splitting each classification where there are two blank lines
    windows = windows.split('\n\n')
    
    # Creating a format for the dataframe, the classification and the 90 measurements across 15 time-steps
    data = {
        'classification': [],
        'window_data': []
    }
    
    # Iterating through each data entry that has been separated to create a single row for each classification
    for window in windows:
        try:
            # Using Regex to find all classifications (searching for values from the alphabet)
            classification = re.findall('^\n?([a-zA-z]+)\n', window)[0]
        except IndexError:
            classification = 'unknown'

        window = window.replace(classification, '')
        # Splitting based on 'tab' value, which is how each measurement is split in the dataset
        window = [line.split('\t') for line in window.split('\n') if line != '']
        # Returning a flattened array of the force and torque values
        window = np.ravel(window)
        # Creating an array of all non-null strings
        window = np.array([v for v in window if v != ''])
        # Ensuring the data-type of the force and torques are integers
        window = window.astype(int)

        data['classification'].append(classification)
        data['window_data'].append(window)
    
    # Naming the columns from 0 - 14 for each force and torque (0 indexed)
    column_names = [
        [f'Fx{i}', f'Fy{i}', f'Fz{i}', f'Tx{i}', f'Ty{i}', f'Tz{i}']
        for i in range(15)
    ]

    # Adding the relevant classification to each row
    column_names = np.ravel(column_names)
    column_names = np.append(column_names, ['label'])
    
    # Creating a list to be converted to pd.DataFrame
    data = [
        np.append([window_data][0], [classification][0])
        for classification, window_data in zip(data['classification'], data['window_data'])
    ]

    # Converting the data into a Pandas dataframe to be used for model building
    df = pd.DataFrame(
        data,
        columns=column_names,
        dtype=np.float32
    )
    
    # Returning the constructed Pandas dataframe
    return df


def graph_observations(df, observation_types, force):
    '''
    This function creates a sub-set of the transformed dataset
    
    Inputs:
      df | pandas.DataFrame | The desired dataset name
      
      observation_types | array | The observation (classification) types for the desired dataset
      
      forces | string | The desired force type to be looked at ('Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz')

    Outputs:
      plt.show() | MatPlotLib.Plt | A graph containing the chosen force type vs time(ms) for the desired dataset
    '''
    
    # Setting the figure size
    fig = plt.figure(figsize=(20, 20))
    
    # Combines the graphs for each classification for easier comparison
    gs = fig.add_gridspec(len(observation_types), 1)

    # Adding an axes to the figure as part of the subplot
    axes = [
        fig.add_subplot(gs[i, 0])
        for i in range(len(observation_types))
    ]

    # Creating an np.linspace of the timesteps, from 0 - 315ms in 15 steps
    observations = 15
    time = np.linspace(0, 315, observations)
    # Setting the colours for each graphing
    colours = [
        'tab:blue', 'tab:orange', 'tab:green',
        'tab:red', 'tab:purple', 'tab:brown',
        'tab:pink', 'tab:gray', 'tab:olive',
        'tab:cyan'
    ]
    
    # Iterating through each ax, observation_type and colour
    for ax, observation_type, colour in zip(axes, observation_types, colours):
        mask = df['label'] == observation_type

        # Setting the values for each force (Fx0, Fx1, Fx2, etc)
        cols = [
            f'{force}{observation}'
            for observation in range(observations)
        ]
        
        # Iterates over dataframe rows as pairs
        for index, row in df[mask].iterrows():
            ax.plot(
                # Defines x-axis
                time,
                # Defines y-axis
                row[cols],
                # Defines colour
                c=colour,
                # Defines opacity
                alpha=0.20
            )
        
        # Ensuring the force type can be displayed
        force_type = force[0]
        subscript = force[1]

        # Setting y-axis label for each graph
        ax.set_ylabel(f'${force_type}_{subscript}$')
        # Setting title for each graph
        ax.set_title(f'{observation_type}: ${force_type}_{subscript}$')

    # Displaying x-axis label
    plt.xlabel('Time (ms)')
    # Displaying the set y-axis label
    plt.ylabel(f'${force_type}_{subscript}$')
    
    # Displaying the created graph
    plt.show()

    
def class_dist(df):
    '''
    This function creates a graph of class distributions for the given dataset
    
    Inputs:
      df | pandas.DataFrame | The desired dataset name. Dataframe should have
      a name. See df.name.
      
    Outputs:
      Seaborn Catplot | sns.catplot | A graph showing distribution of classes for the dataset
    '''

    df_class = pd.crosstab(
        df['label'],
        columns='count'
    )
    
    df_class = df_class.reset_index()
    df_class = df_class.sort_values('count', ascending=False)
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Plots a seaborn catplot of class distributions
    sns.barplot(
        data=df_class,
        x='label', y='count',
        ax=ax
    )
    
    # Setting figure title
    ax.set_title(f'Class Distribution for {df.name}')
    # Rotating x-axis labels to ensure readability
    plt.xticks(rotation=40)
    
    
def build_model(
    df, input_cols, output_cols,
    classifier, hyperparams,
    cv, n_iter,
    smote=True, pca=0,
    verbose=True,
):
    '''
    This function automatically finds the best hyperparameters for the chosen
    classification method, using RandomizedSearchCV from SKLearn. Prints a classification report
    and SCORES values for the best model.
    
    Inputs:
      df | pandas.DataFrame | The desired dataset name
      
      input_cols | array | The input column names
      
      output_cols | array | The output column names
      
      classifier | classifier | The desired classifier to use
      
      hyperparams | dictionary | The desired range of hyperparamaters to use
      
      cv | int | The desired number of cross folds, if cv=1 no cross folds are used
      
      n_iter | int | The number of iterations for RandomizedSearchCV to run
      
      smote | boolean | Boolean value defining if SMOTE should be used
      
      verbose | boolean | Boolean value defining if verbose should be displayed
      
    Outputs:
      report | dictionary | Dictionary containing the classification report from RandomizedSearchCV
      
      scores | pd.DataFrame | Pandas Dataframe containing the score values for the classifier
    '''
    
    # Copying the model, prevents it
    df_model = df.copy()
    
    #
    X = df_model[input_cols]
    #
    y = df_model[output_cols].copy()
    
    #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.4,
        random_state=0
    )

    if smote:
        sm = SMOTE(
            random_state=0
        )

        X_train, y_train = sm.fit_resample(
            X_train,
            y_train.values
        )

    if cv == 1:
        search = RandomizedSearchCV(
            # Feeding the constructed pipeline in
            classifier,
            # Feeding in the defined hyperparameters
            hyperparams,
            # Defining scorer to use
            scoring='accuracy',
            # Number of parameter settings sampled
            n_iter=n_iter,
            # Using all processors
            n_jobs=-1,
            # Setting CV value to ShuffleSplit with 1 split
            # RandomizedSearchCV does not allow cv=1
            cv=ShuffleSplit(n_splits=1, random_state=0),
            verbose=verbose,
            # Setting random state=0 for reproducibility
            random_state=0
        )
        
    else:
        search = RandomizedSearchCV(
            classifier,
            hyperparams,
            scoring='accuracy',
            n_iter=n_iter,
            n_jobs=-1,
            # Uses Stratified K-folds
            cv=cv,
            verbose=verbose,
            random_state=0
        )
    
    search = search.fit(X=X_train, y=y_train)

    y_pred = search.best_estimator_.predict(X_test)
    print('Best Hyperparameters: ')
    print(search.best_estimator_)
    
    labels = ['accuracy', 'f1', 'precision', 'roc_auc', 'recall', 'specificity']
#     scores = {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'balanced accuracy': balanced_accuracy_score(y_test, y_pred),
#         'precision': ,
#         'roc_auc': ,
#         'recall': recall_score(y_test, y_pred),
#     }
    
    if len(output_cols) > 1:
        target_names = output_cols
        conf_matrix = multilabel_confusion_matrix(y_test, y_pred, labels=target_names)
        for tn, fp, fn, tp in conf_matrix:
            tn, fp, fn, tp = conf_matrix[]
        scores = {
            'accuracy': [accuracy_score(y_test, y_pred, labels=target_names)],
            'balanced accuracy': [balanced_accuracy_score(y_test, y_pred, labels=target_names)],
            'precision': [precision_score(y_test, y_pred, average='micro', labels=target_names)],
            'recall': [recall_score(y_test, y_pred, average='micro', labels=target_names)],
            'roc_auc': [roc_auc_score(y_test, y_pred, average='micro', labels=target_names)],
            'F1': [f1_score(y_test, y_pred, average='micro', labels=target_names)],
#             'specificity': conf_matrix[]
        }
    elif len(output_cols) == 1:
        target_names = ['Normal', 'Failure']
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        scores = {
            'accuracy': [accuracy_score(y_test, y_pred)],
            'balanced accuracy': [balanced_accuracy_score(y_test, y_pred)],
            'precision': [precision_score(y_test, y_pred)],
            'recall': [recall_score(y_test, y_pred)],
            'roc_auc': [roc_auc_score(y_test, y_pred)],
            'F1': [f1_score(y_test, y_pred)],
            'specificity': [tn / (tn + fp)]
        }
    # CHANGE TO SCORES
    scores = pd.DataFrame(scores)
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=1
    )
    print(report)
    print(scores)
    return search.best_estimator_