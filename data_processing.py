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
    multilabel_confusion_matrix, ConfusionMatrixDisplay,
    auc, roc_curve
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
    smote=True, test_size=0.3,
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
      
      test_size | float | Ratio to use for the train/test split
      
      verbose | boolean | Boolean value defining if verbose should be displayed
      
    Outputs:
      report | dictionary | Dictionary containing the classification report from RandomizedSearchCV
      
      scores | pd.DataFrame | Pandas Dataframe containing the score values for the classifier
      
      
    '''
    
    # Copying the model, prevents it
    df_model = df.copy()
    
    # Creating input columns
    X = df_model[input_cols].copy()
    # Creating output columns
    y = df_model[output_cols].copy()
    
    # Defining Test/Train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=0
    )
    
    # Using SMOTE if smote=True
    if smote:
        sm = SMOTE(
            random_state=0,
            k_neighbors=1
        )

        X_train, y_train = sm.fit_resample(
            X_train,
            y_train.values
        )
    
    # No Cross-folds
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
    # Cross-folds included    
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
    
    # Fitting the RandomizedSearchCV to the training dataset
    search = search.fit(X=X_train, y=y_train)

    # Using the best estimator from RandomizedSearchCV to predict test dataset
    y_pred = search.best_estimator_.predict(X_test)
    print('Best Hyperparameters: ')
    print(search.best_estimator_.get_params())
    print()
    
    # Multi-class case
    if len(output_cols) > 1:
        # Defining target names
        target_names = output_cols
        
        # Creating a multi-class confusion matrix
        conf_matrix = confusion_matrix(
            y_test.values.argmax(axis=1),
            search.best_estimator_.predict(X_test).argmax(axis=1), 
        )
        
        # Creating empty dictionary to store score metrics
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'auc': [],
            'F1': []
        }
        
        # Calculating the accuracy from the confusion matrix
        accuracy = np.diag(conf_matrix).sum() / conf_matrix.sum()
        
        scores['accuracy'].append(accuracy)
        scores['precision'].append(
            precision_score(y_test, y_pred, average='weighted', zero_division=0)
        )
        scores['recall'].append(
            recall_score(y_test, y_pred, average='weighted', zero_division=0)
        )
        scores['auc'].append(
            roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')
        )
        scores['F1'].append(
            f1_score(y_test, y_pred, average='weighted')
        )
    
        # Converting dictionary to pd.DataFrame
        scores = pd.DataFrame(scores)
        
        # Plotting the confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix,
            display_labels=output_cols,
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.grid(False)
        disp.plot(cmap='Blues', ax=ax)
        plt.title('Confusion Matrix')
        plt.xticks(rotation=90)
        plt.show()
        
        # Creating dictionaries of fpr, tpr, roc_auc
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        # Iterating through the output of the dataframe
        for i, output in enumerate(output_cols):
            fpr[output], tpr[output], _ = roc_curve(
                y_test.values[:, i],
                y_pred[:, i]
            )
            # Calculating AUC
            roc_auc[output] = auc(fpr[output], tpr[output])

        # Compute weighted-average ROC curve and AUC
        fpr["weighted"], tpr["weighted"], _ = roc_curve(y_test.values.ravel(), y_pred.ravel())
        roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])
        
        plt.figure(figsize=(10, 10))
        
        # Iterating through outputs, plotting ROC
        for output in list(output_cols) + ['weighted']:
            plt.plot(
                fpr[output], tpr[output],
                label=f'{output} AUC: {round(roc_auc[output], 3)}'
            )

        # Plotting the Base Rate ROC
        plt.plot([0,1], [0,1],label='Base Rate')
        # Setting x & y limits for graph
        plt.xlim([-0.005, 1.0])
        plt.ylim([0.0, 1.05])
        # Setting axis labels, title & legend
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Graph')
        plt.legend(loc="lower right")
        plt.show()
        
    # Binary-class case
    elif len(output_cols) == 1:
        target_names = ['Failure', 'Normal']
        
        # Setting empty dict of score metrics
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'auc': [],
            'F1': [],
            'specificity': []
        }
        
        # Creating array of positive/neg values
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Appending calculated metrics
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        scores['auc'].append(roc_auc_score(y_test, y_pred, average='weighted'))
        scores['F1'].append(f1_score(y_test, y_pred, average='weighted'))
        scores['specificity'].append(tn / (tn + fp))
        
        # Plotting the confusion matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.grid(False)
        plot_conf_matrix = plot_confusion_matrix(
            search.best_estimator_, X_test, y_test,
            display_labels=target_names, cmap=plt.cm.Blues,
            ax=ax
        )
        plt.title('Confusion Matrix')
        plt.show()
        
        # Creating roc_curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        
        plt.figure(figsize=(10, 10))
        
        # Calculating AUC
        auc_score = auc(fpr, tpr)

        # Plot Random Forest ROC
        plt.plot(fpr, tpr, label=f'AUC: {round(auc_score, 3)}')

        # Plot Base Rate ROC
        plt.plot([0,1], [0,1],label='Base Rate')
        # Setting x & y limits for graph
        plt.xlim([-0.005, 1.0])
        plt.ylim([0.0, 1.05])
        # Setting axis labels, title & legend
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Graph')
        plt.legend(loc="lower right")
        plt.show()
        
    # Converting scores to pd.DataFrame
    scores = pd.DataFrame(scores)
    # Creating a classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0
    )
    
    print(report)
    print(scores)
    print()
    print(f'Time to fit best model: {search.refit_time_} seconds')
    return search.best_estimator_