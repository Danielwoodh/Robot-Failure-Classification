# Robot Failure Classification
This project is an assessment for Cranfield University involving the construction of machine-learning models to classify datasets containing Force and Torque data relating to an operation type of a robot.

## Files contained

The jupyter notebook 'robot-failure-classifcation.ipynb' shows the construction of all of the ML models as well as the data preparation. It also shows a demonstration of how SMOTE can be used to fix some of the class imbalance issues in the dataset. A demonstration of PCA for LP1 is shown but not implemented.

The 'data_processing.py' file contains the main functions that are called within the 'robot-failure-classifcation.ipynb' notebook.

The 'requirements.txt' contains a list of all the necessary libraries for the notebook to run, these MUST be installed prior to attempting to run the notebook.

The 'Dataset' folder contains the datasets that are analysed, LP1-5.

The 'LICENSE' file is used only for uploading to Github.

The '350272-Report.pdf' file contains the report that was created, outlining the literature review, methodology, and results that were produced to accompany this project.

## Installation

Make sure you are running Python 3.7 or above.

Use the package manager *pip* to install the necessary libraries. It is recommended that all packages are using their laters versions.

To install Jupyter Lab, run this command in the Command Prompt:

```cmd
pip install jupyterlab
```

In the Command Prompt, enter this command to install the necessary libraries:

```cmd
pip install --user --upgrade -r requirements.txt
```

### Usage
---
To run the Jupyter Notebook file, run this command in the Command Prompt:

```cmd
jupyter lab
```

If this command does not work, try the following:

```cmd
jupyter-lab
```

```cmd
jupyter notebook
```

Once Jupyter Lab has opened, open the 'robot-failure-classifcation.ipynb' file.
