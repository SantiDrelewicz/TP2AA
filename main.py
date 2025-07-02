import pandas as pd
import torch
import random
import argparse

from classic import ClassicPunctuationCapitalizationModel, evaluate_model
from rnn import RNNPunctuationCapitalizationModel, evaluate_model_rnn
from utils import split_data_from_file

RANDOM_SEED = 0

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def main(mode: str, which: str):
    train_sentences, val_sentences, test_sentences = split_data_from_file(
        "es_419_validas.txt",
        train_ratio=0.8,
        val_ratio=0.1,
        max_lines=80000
    )

    if mode == "train":
        print(f"Training on {len(train_sentences)} instances, validating on {len(val_sentences)} instances")

        if which in ("classic", "both"):
            classic_model = ClassicPunctuationCapitalizationModel()
            classic_model.train(train_sentences)
            classic_model.save_model("classic_punct_model.pkl")

        if which in ("rnn", "birnn", "both"):
            bidirectional = (which == "birnn")
            rnn_model = RNNPunctuationCapitalizationModel(bidirectional=bidirectional)
            rnn_model.train(train_sentences, val_sentences, epochs=10)
            filename = "trained_birnn_model.pt" if bidirectional else "trained_rnn_model.pt"
            rnn_model.save_model(filename)


    elif mode == "test":
        print(f"Testing on {len(test_sentences)} instances")

        if which in ("classic", "both"):
            classic_model = ClassicPunctuationCapitalizationModel()
            classic_model.load_model("classic_punct_model.pkl")
            a = evaluate_model(classic_model, classic_model._prepare_data(test_sentences))
            print(classic_model.predict_and_reconstruct("pasado mañana"))
            print(classic_model.predict_and_reconstruct("estás asustado"))
            print(classic_model.predict_and_reconstruct("cindy espero que estes muy orgullosa de lo que haz hecho"))
            print(classic_model.predict_and_reconstruct("cómo estás"))

        if which in ("rnn", "birnn", "both"):
            bidirectional = (which == "birnn")
            rnn_model = RNNPunctuationCapitalizationModel(bidirectional=bidirectional)
            filename = "trained_birnn_model.pt" if bidirectional else "trained_rnn_model.pt"
            rnn_model.load_model(filename)

            # evaluate_model_rnn(rnn_model, test_sentences)
            # print(rnn_model.predict_and_reconstruct("pasado mañana"))
            # print(rnn_model.predict_and_reconstruct("estás asustado"))
            # print(rnn_model.predict_and_reconstruct("cindy espero que estes muy orgullosa de lo que haz hecho"))

            input_csv = "datos_test.csv"
            output_csv = "predicted_output.csv"

            input_df = pd.read_csv(input_csv)
            rnn_model.predict_to_csv_from_dataframe(input_df, output_file=output_csv)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'test'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test punctuation-capitalization models")
    parser.add_argument(
        "mode",
        choices=["train", "test"],
        help="Mode: train models or test existing models"
    )
    parser.add_argument(
        "--model",
        choices=["classic", "rnn", "birnn", "both"],
        default="both",
        help="Which model(s) to train/test (default: both)"
    )
    args = parser.parse_args()
    main(args.mode, args.model)
