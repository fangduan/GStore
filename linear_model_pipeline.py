import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns

def linear_model_pipeline(X_train, X_test, y_train, y_test):
    # metric MAPE
    def MAPError(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean((np.abs(y_true - y_pred)) / (y_true)) * 100



    # Function to fit the regressor and record performance metrics
    def pipeline(reg, X_train, y_train, X_test, y_test, **kwargs):
        # Dictionary to hold properties of Models
        reg_props = {}

        # Initialize and fit the regressor, and time taken
        regressor = reg(**kwargs)
        start = time()
        regressor.fit(X_train, y_train)
        end = time()

        # Store the metrics for the regressor
        reg_props["name"] = reg.__name__
        reg_props["train_time"] = end - start
        reg_props["train_score"] = regressor.score(X_train, y_train)
        reg_props["test_score"] = regressor.score(X_test, y_test)
        reg_props["rmse"] = np.sqrt(mean_squared_error(y_test, regressor.predict(X_test)))
        reg_props["MAE"] = mean_absolute_error(y_test, regressor.predict(X_test))
        reg_props["MAPE"] = MAPError(y_test, regressor.predict(X_test))

        return reg_props



    # Function to execute each algorithm through the pipeline
    def execute_pipeline():
        # Create the list of algorithms
        regressors = [
            LinearRegression,
            Ridge,
            KNeighborsRegressor,
            RandomForestRegressor,
            GradientBoostingRegressor,
            MLPRegressor,
            ExtraTreesRegressor,
        ]

        # To store the properties for each regressor
        props = []

        """
        Iterate thorugh the list of regressors,
        passing each thorugh the pipeline and
        storing its properites
        """
        for reg in regressors:
            properites = pipeline(reg, X_train, y_train, X_test, y_test)
            props.append(properites)

        return props

    def get_properties():
        # Obtain the properties after executing the pipeline
        properties = execute_pipeline()

        # Extract each individual property of the Regressors
        names = [prop["name"] for prop in properties]
        train_times = [prop["train_time"] for prop in properties]
        train_scores = [prop["train_score"] for prop in properties]
        test_scores = [prop["test_score"] for prop in properties]
        rmse_vals = [prop["rmse"] for prop in properties]
        mae_vals = [prop["MAE"] for prop in properties]
        mape_vals = [prop["MAPE"] for prop in properties]

        # Create a DataFrame from these properties
        df = pd.DataFrame(index=names,
                          data={
                              "Training Times": train_times,
                              "Training Scores": train_scores,
                              "Testing Scores": test_scores,
                              "RMSE": rmse_vals,
                              "MAE": mae_vals,
                              "MAPE": mape_vals
                          }
                          )

        return df

    df = get_properties()
    return df





def plot_performance(properties):
    # Plot to compare thePerformance of Algorithms
    sns.set_context("notebook", font_scale=1.7)
    plt.figure(figsize=(18, 7))
    plt.subplot(2, 2, 1)
    plt.ylabel("RMSE OF Regressors")
    properties["RMSE"].plot(kind="barh", color='#4d80b3');
    sns.despine(left=True)
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.ylabel("MAE OF Regressors")
    properties["MAE"].plot(kind="barh", color='#4d80b3');
    sns.despine(left=True)
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.ylabel("MAPE OF Regressors")
    properties["MAPE"].plot(kind="barh", color='#4d80b3');
    sns.despine(left=True)
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.ylabel("Training Time of Regressors")
    properties["Training Times"].plot(kind="barh", color='#4d80b3');
    sns.despine(left=True)
    plt.tight_layout()


def plot_compare_performanc(properties):
    # Plot to compare the performance of Models on both datasets
    sns.set_style('white')
    sns.set_context("paper", font_scale=1.3)

    ax = properties[["MAE", "RMSE"]].plot(kind="barh", color=['#9491B5', '#685CF4'], title="Performance of Regressor",
                                          figsize=(10, 4))
    sns.despine(left=True);
    plt.tight_layout()
    plt.show()


properties = linear_model_pipeline(X_train, X_test, y_train, y_test)
plot_performance(properties)
plot_compare_performanc(properties)