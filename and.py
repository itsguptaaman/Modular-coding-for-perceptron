import pandas as pd
from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import logging
import os


GATE = "AND"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "perceptron_logs.log"),
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s: %(module)s]: %(message)s',
    filemode='a'
)


def main(data, modelName, plotName, eta, epochs):
    df_and = pd.DataFrame(data)
    logging.info(f"This is the raw dataset: \n{df_and}")
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

    try:
        logging.info(f">>>> Starting the training for {GATE} <<<<")
        main(data=AND, modelName="and_model.pkl",
             plotName="and.png", eta=ETA, epochs=EPOCHS)
        logging.info(f">>>> Ending trainging for {GATE} <<<<")

    except Exception as e:
        logging.exception(e)
        raise e
