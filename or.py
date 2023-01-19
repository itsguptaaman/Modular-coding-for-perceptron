import pandas as pd
from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron


def main(data, modelName, plotName, eta, epochs):
    df_or = pd.DataFrame(data)
    X, y = prepare_data(df_or)

    model = Perceptron(learning_rate=eta, epochs=epochs)

    model.fit(X, y)

    # _ Place_holder is a dummy variable
    _ = model.total_loss()
    model.model_store(file_name=modelName)

    model_or = Perceptron().model_load(file_path="Model/or_model.pkl")
    save_plot(df_or, model_or, file_name=plotName)


if __name__ == '__main__':
    OR = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 1, 1, 1]
    }

    # learning Rate (eta) 0 to 1
    ETA = 0.1
    EPOCHS = 10

    main(data=OR, modelName="or_model.pkl",
         plotName="or.png", eta=ETA, epochs=EPOCHS)
