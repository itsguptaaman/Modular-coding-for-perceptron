import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import seaborn as sns
import joblib
from matplotlib.colors import ListedColormap
import warnings

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")  # << Optional >>

# Creating a function to segrigate data


def prepare_data(df, target_column="y"):
    """This function will prepare data (Dependent and Independent features)

    Args:
        df (pd.DataFrame): This is a Data Frame
        target_column (str, optional): Label column name is given Defaults to "y".

    Returns:
        Tuple: returns Labels and X
    """
    logging.info(">>>> Prepraing data for training <<<<")
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return X, y


def save_plot(df, model, file_name="plot.png", plot_dir="plots"):

    def _create_base_plot(df):
        logging.info(">>>> Creating the base plot <<<<")
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="coolwarm")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)

        figure = plt.gcf()
        figure.set_size_inches(10, 8)

    def _plot_decision_regions(X, y, classifier, resolution=0.02):
        logging.info(">>>> Ploting the decisions info <<<<")
        colors = ("cyan", "lightgreen")
        cmap = ListedColormap(colors)

        X = X.values  # As an array
        x1 = X[:, 0]
        x2 = X[:, 1]

        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1

        xx1, xx2, = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                np.arange(x2_min, x2_max, resolution)
                                )

        y_hat = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        y_hat = y_hat.reshape(xx1.shape)

        plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        plt.plot()

    X, y = prepare_data(df)
    _create_base_plot(df)
    _plot_decision_regions(X, y, model)

    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, file_name)
    plt.savefig(plot_path)
    logging.info(f">>>> Saving the plot at {plot_path} <<<<")
