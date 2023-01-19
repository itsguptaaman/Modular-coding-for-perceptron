import pandas as pd
from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron

def main(data, modelName, plotName, eta, epochs):
    df_and = pd.DataFrame(data)
    X, y = prepare_data(df_and)

    model = Perceptron(learning_rate=eta, epochs=epochs)

    model.fit(X, y)

    # _ Place_holder is a dummy variable
    _ = model.total_loss()
    model.model_store(file_name=modelName)

    model_and = Perceptron().model_load(file_path="Model/and_model.pkl")
    save_plot(df_and, model_and, file_name=plotName)

if __name__ == '__main__':
    AND = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 0, 0, 1]
    }

    

    # learning Rate (eta) 0 to 1
    ETA = 0.1
    EPOCHS = 10

    main(data=AND, modelName="and_model.pkl", plotName="and.png", eta=ETA, epochs=EPOCHS)
