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
    roc_auc_score, classification_report
)

# SMOTE
from imblearn.over_sampling import SMOTE

# Pipeline
from imblearn.pipeline import Pipeline


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
      df | pandas.DataFrame | The desired dataset name
      
    Outputs:
      Seaborn Catplot | sns.catplot | A graph showing distribution of classes for the dataset
    '''
    
    # Receives the count of all classes in the dataset & converts to a pd.DataFrame
    df_class = df['label'].value_counts().to_frame()
    # Changes the classes to index names
    df_class.index.name = 'class'
    # Resets the index values
    df_class.reset_index(inplace=True)
    
    # Plots a seaborn catplot of class distributions
    a = sns.catplot(
            data=df_class, kind='bar',
            x='class', y='label'
    )
    
    # Setting figure title
    a.fig.suptitle(f'Class Distribution for {df.name}')
    # Setting figure axis labels
    a.set_axis_labels('Class', 'Frequency')
    # Rotating x-axis labels to ensure readability
    a.set_xticklabels(rotation=40, ha="right")
    
    
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
      
      pca | boolean | Boolean value defining if PCA should be used
      
      verbose | boolean | Boolean value defining if verbose should be displayed
      
    Outputs:
      report | dictionary | Dictionary containing the classification report from RandomizedSearchCV
      
      scores | dictionary | Dictionary containing the score values for the classifier
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
        test_size=0.3,
        random_state=0
    )
    
    # NEED TO APPLY SMOTE ONLY TO TRAIN SET, HOW???
    
    # Logic statements determining which pipeline to use based on given values
    if smote == True and pca >= 1:
        pipe = Pipeline([
            ('sampling', SMOTE()),
            ('standardisation', StandardScaler()),
            ('pca', PCA(n_components=18))
            ('classifier', classifier(random_state=0))
        ])
    elif smote == True:
        pipe = Pipeline([
            ('sampling', SMOTE()),
            ('classifier', classifier(random_state=0))
        ])
    elif pca >= 1:
        pipe = Pipeline([
            ('standardisation', StandardScaler()),
            ('pca', PCA(n_components=18)),
            ('classifier', classifier(random_state=0))
        ])
    else:
         pipe = classifier(random_state=0)
    
    if cv == 1:
        search = RandomizedSearchCV(
            # Feeding the constructed pipeline in
            pipe,
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
            # 
            verbose=verbose,
            # Setting random state=0 for reproducibility
            random_state=0
        )
        
    else:
        search = RandomizedSearchCV(
            pipe,
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
    
#     labels = ['accuracy', 'f1', 'precision', 'roc_auc', 'recall', 'specificity']
#     scores = {
#         '': 
#         '':
#         '':
#     }
    print()
    
    # CHANGE TO SCORES
    
    report = classification_report(y_test, y_pred)
    print(report)
    
    return search.best_estimator_