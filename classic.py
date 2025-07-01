import pandas as pd
import numpy as np
import pickle

from scipy import sparse

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report

from transformers import BertTokenizer

from collections import defaultdict

from utils import extract_labels

PUNCT_INITIAL_LABELS = {"": 0, "¿": 1}
PUNCT_INITIAL_LABELS_INV = {v: k for k, v in PUNCT_INITIAL_LABELS.items()}

PUNCT_FINAL_LABELS = {"": 0, ",": 1, ".": 2, "?": 3}
PUNCT_FINAL_LABELS_INV = {v: k for k, v in PUNCT_FINAL_LABELS.items()}

class FeatureExtractor:
    def __init__(self):
        self.word_freq = defaultdict(int)
        self.pos_patterns = {}

    def fit(self, dataset):
        """
        Learn patterns from all training data
        """
        for instance in dataset:
            tokens = instance["tokens"]
            capitalization = instance["capitalization"]
            for i, token in enumerate(tokens):
                clean_token = token.replace("##", "")
                self.word_freq[clean_token] += 1
                if capitalization[i] > 0:
                    self.pos_patterns[clean_token.lower()] = capitalization[i]

    def extract_features(self, tokens, position):
        """Extract features for a token at given position"""
        token = tokens[position]
        clean_token = token.replace("##", "")

        features = {
            # Token features
            "token_length": len(clean_token),
            "is_subtoken": token.startswith("##"),
            "has_apostrophe": "'" in token,
            "is_numeric": clean_token.isdigit(),
            "word_freq": self.word_freq.get(clean_token.lower(), 0),
            # Position features
            "is_first": position == 0,
            "is_last": position == len(tokens) - 1,
            "position_ratio": position / len(tokens),
            # Context features
            "prev_token": tokens[position - 1] if position > 0 else "<START>",
            "next_token": (
                tokens[position + 1] if position < len(tokens) - 1 else "<END>"
            ),
            # Pattern features
            "known_capitalization": self.pos_patterns.get(clean_token.lower(), 0),
            "is_question_word": clean_token.lower()
            in ["qué", "cuándo", "dónde", "cómo", "por", "quién"],
        }

        # Add bigram features
        if position > 0:
            prev_clean = tokens[position - 1].replace("##", "")
            features["prev_bigram"] = f"{prev_clean}_{clean_token}"

        if position < len(tokens) - 1:
            next_clean = tokens[position + 1].replace("##", "")
            features["next_bigram"] = f"{clean_token}_{next_clean}"

        return features


class ClassicPunctuationCapitalizationModel:
    def __init__(self):
        print("initiating...")
        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
        self.feature_extractor = FeatureExtractor()

        # Separate models for each task
        self.punct_initial_model = RandomForestClassifier(
            n_estimators=80, random_state=0, verbose=0
        )
        self.punct_final_model = RandomForestClassifier(
            n_estimators=80, random_state=0, verbose=0
        )
        self.capitalization_model = RandomForestClassifier(
            n_estimators=80, random_state=0, verbose=0
        )

        # Vectorizers for categorical features
        self.vectorizers = {
            "prev_token": HashingVectorizer(n_features=256, alternate_sign=False),
            "next_token": HashingVectorizer(n_features=256, alternate_sign=False),
            "prev_bigram": HashingVectorizer(n_features=128, alternate_sign=False),
            "next_bigram": HashingVectorizer(n_features=128, alternate_sign=False),
        }

        self.is_fitted = False

    def _prepare_data(self, raw_sentences):
        data = []
        instances = []

        for inst_id, sentence in enumerate(raw_sentences, start=1):
            words, init_lbls, final_lbls, cap_lbls = extract_labels(sentence)
            token_idx = 0
            inst_data = []

            for word, init_lbl, final_lbl, cap_lbl in zip(words, init_lbls, final_lbls, cap_lbls):
                subtokens = self.tokenizer.tokenize(word.lower())
                for i, sub in enumerate(subtokens):
                    punct_init = init_lbl if i == 0 else ""
                    punct_final = final_lbl if i == len(subtokens) - 1 else ""
                    inst_data.append([inst_id, token_idx, sub, punct_init, punct_final, cap_lbl])
                    token_idx += 1

            data.extend(inst_data)

            if inst_id % 50_000 == 0:
                print(f"… processed {inst_id} sentences")
                
        df = pd.DataFrame(data, columns=["inst_id", "token_id", "token", "punt_inicial", "punt_final", "capitalizacion"])

        grouped = {}
        for inst_id, group in df.groupby("inst_id"):
            grouped[inst_id] = {
                "tokens": group["token"].tolist(),
                "punct_initial": group["punt_inicial"].tolist(),
                "punct_final": group["punt_final"].tolist(),
                "capitalization": group["capitalizacion"].tolist(),
            }

        return list(grouped.values())

    def prepare_features(self, feature_dicts, fit_vectorizers=False):
        """Convert feature dictionaries to numerical arrays, keeping sparse matrices."""
        categorical_features = [
            "prev_token",
            "next_token",
            "prev_bigram",
            "next_bigram",
        ]
        numerical_features = [
            k for k in feature_dicts[0].keys() if k not in categorical_features
        ]

        # Numerical features: dense
        X_numerical = np.array(
            [[fd.get(feat, 0) for feat in numerical_features] for fd in feature_dicts]
        )

        # Convert to sparse for hstack
        X_numerical_sparse = sparse.csr_matrix(X_numerical)

        # Categorical features: keep sparse
        X_categorical_parts = []
        for feat in categorical_features:
            feat_values = [fd.get(feat, "") for fd in feature_dicts]

            if fit_vectorizers:
                X_feat = self.vectorizers[feat].fit_transform(feat_values)
            else:
                X_feat = self.vectorizers[feat].transform(feat_values)

            X_categorical_parts.append(X_feat)

        # Combine all
        if X_categorical_parts:
            X_all = sparse.hstack([X_numerical_sparse] + X_categorical_parts)
        else:
            X_all = X_numerical_sparse

        return X_all


    def tokenize_and_get_ids(self, text):
        """Tokenize text and return tokens with their IDs"""
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_ids

    def train(self, raw_sentences):
        training_data = self._prepare_data(raw_sentences)

        all_features = []
        all_punct_initial = []
        all_punct_final = []
        all_capitalization = []

        self.feature_extractor.fit(training_data)

        print("extracting token data")
        for instance in training_data:
            tokens = instance["tokens"]
            punct_initial = instance["punct_initial"]
            punct_final = instance["punct_final"]
            capitalization = instance["capitalization"]

            for i in range(len(tokens)):
                features = self.feature_extractor.extract_features(tokens, i)
                all_features.append(features)
                all_punct_initial.append(punct_initial[i])
                all_punct_final.append(punct_final[i])
                all_capitalization.append(capitalization[i])

        X = self.prepare_features(all_features, fit_vectorizers=True)
        all_punct_initial_num = [PUNCT_INITIAL_LABELS.get(lbl, 0) for lbl in all_punct_initial]
        all_punct_final_num = [PUNCT_FINAL_LABELS.get(lbl, 0) for lbl in all_punct_final]
        all_capitalization_num = [int(c) for c in all_capitalization]

        print("fitting models...")

        print("fitting initial model")
        self.punct_initial_model.fit(X, all_punct_initial_num)

        print("fitting final model")
        self.punct_final_model.fit(X, all_punct_final_num)

        print("fitting capitalization model")
        self.capitalization_model.fit(X, all_capitalization_num)

        self.is_fitted = True


    def predict(self, text):
        """Predict punctuation and capitalization for input text"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        tokens, token_ids = self.tokenize_and_get_ids(text)

        # Extract features
        features = []
        for i in range(len(tokens)):
            feat_dict = self.feature_extractor.extract_features(tokens, i)
            features.append(feat_dict)

        # Convert to numerical format
        X = self.prepare_features(features, fit_vectorizers=False)

        # Make predictions (numerical labels)
        punct_initial_pred = self.punct_initial_model.predict(X)
        punct_final_pred = self.punct_final_model.predict(X)
        capitalization_pred = self.capitalization_model.predict(X)

        # Decode labels back to strings/ints
        punct_initial_pred_decoded = [PUNCT_INITIAL_LABELS_INV.get(p, "") for p in punct_initial_pred]
        punct_final_pred_decoded = [PUNCT_FINAL_LABELS_INV.get(p, "") for p in punct_final_pred]
        capitalization_pred_decoded = [int(p) for p in capitalization_pred]

        # Optional: unify capitalization predictions across subtokens of same word
        capitalization_pred_decoded = unify_subtoken_caps(tokens, capitalization_pred_decoded)

        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "punct_initial": punct_initial_pred_decoded,
            "punct_final": punct_final_pred_decoded,
            "capitalization": capitalization_pred_decoded,
        }

    def predict_to_csv(self, text, instance_id=1, output_file=None):
        """Predict and format as CSV"""
        predictions = self.predict(text)

        # Create DataFrame
        df_data = []
        for i, (token, token_id, pi, pf, cap) in enumerate(
            zip(
                predictions["tokens"],
                predictions["token_ids"],
                predictions["punct_initial"],
                predictions["punct_final"],
                predictions["capitalization"],
            )
        ):
            df_data.append(
                {
                    "instancia_id": instance_id,
                    "token_id": token_id,
                    "token": token,
                    "punt_inicial": pi,
                    "punt_final": pf,
                    "capitalización": cap,
                }
            )

        df = pd.DataFrame(df_data)

        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")

        return df

    def predict_and_reconstruct(self, text):
        result = self.predict(text)
        return self.reconstruct_sentence(
            result["tokens"],
            result["punct_initial"],
            result["punct_final"],
            result["capitalization"]
        )

    def reconstruct_sentence(self, tokens, punct_initial, punct_final, capitalization):
        words = []
        i = 0
        while i < len(tokens):
            if not tokens[i].startswith("##"):
                # Collect subtokens
                word_tokens = [tokens[i].replace("##", "")]
                j = i + 1
                while j < len(tokens) and tokens[j].startswith("##"):
                    word_tokens.append(tokens[j][2:])
                    j += 1
                word = "".join(word_tokens)

                # Apply capitalization
                cap = capitalization[i]
                if cap == 3:
                    word = word.upper()
                elif cap == 1:
                    word = word.capitalize()
                elif cap == 2:
                    if len(word) > 1:
                        word = word[0].upper() + word[1:]
                    else:
                        word = word.upper()
                else:
                    word = word.lower()

                # Attach final punctuation
                if punct_final[i]:
                    word = word + punct_final[i]

                # Prepend initial punctuation
                if punct_initial[i]:
                    word = punct_initial[i] + word

                words.append(word)
                i = j
            else:
                i += 1

        return " ".join(words)

    def save_model(self, filepath):
        """Save trained model"""
        model_data = {
            "feature_extractor": self.feature_extractor,
            "punct_initial_model": self.punct_initial_model,
            "punct_final_model": self.punct_final_model,
            "capitalization_model": self.capitalization_model,
            "vectorizers": self.vectorizers,
            "is_fitted": self.is_fitted,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.feature_extractor = model_data["feature_extractor"]
        self.punct_initial_model = model_data["punct_initial_model"]
        self.punct_final_model = model_data["punct_final_model"]
        self.capitalization_model = model_data["capitalization_model"]
        self.vectorizers = model_data["vectorizers"]
        self.is_fitted = model_data["is_fitted"]

        self.punct_initial_model.verbose = False
        self.punct_final_model.verbose = False
        self.capitalization_model.verbose = False

        print(f"Model loaded from {filepath}")


def create_sample_training_data():
    """Create sample training data for testing"""
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

    training_examples = [
        {
            "text": "cuándo vamos a mcdonald's ellos no vienen hoy dónde están ahora",
            "punct_initial": [
                "¿",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "¿",
                "",
                "",
                "",
            ],
            "punct_final": [
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "?",
                "",
                "",
                "",
                "",
                ".",
                "",
                "",
                "",
                "?",
            ],
            "capitalization": [
                1,
                1,
                1,
                0,
                0,
                0,
                2,
                2,
                2,
                2,
                2,
                2,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
            ],
        },
        {
            "text": "hola cómo estás bien y tú",
            "punct_initial": ["", "¿", "", "", "", "", ""],
            "punct_final": ["", "", "", "?", "", "", "."],
            "capitalization": [1, 0, 0, 0, 0, 0, 0],
        },
    ]

    formatted_data = []
    for example in training_examples:
        tokens = tokenizer.tokenize(example["text"])
        formatted_data.append(
            {
                "tokens": tokens,
                "punct_initial": example["punct_initial"][: len(tokens)],
                "punct_final": example["punct_final"][: len(tokens)],
                "capitalization": example["capitalization"][: len(tokens)],
            }
        )

    return formatted_data

def aggregate_word_level(tokens, labels):
    word_labels = []
    i = 0
    while i < len(tokens):
        current_label = labels[i]
        # Include all subtokens for the current word
        j = i + 1
        while j < len(tokens) and tokens[j].startswith("##"):
            # For labels like capitalization, take max or majority
            current_label = max(current_label, labels[j])
            j += 1
        word_labels.append(current_label)
        i = j
    return word_labels

def evaluate_model(model, test_data):
    all_true_pi, all_pred_pi = [], []
    all_true_pf, all_pred_pf = [], []
    all_true_cap, all_pred_cap = [], []

    for instance in test_data:
        text = model.tokenizer.convert_tokens_to_string(instance["tokens"])
        predictions = model.predict(text)

        true_pi_enc = [PUNCT_INITIAL_LABELS.get(lbl, 0) for lbl in instance["punct_initial"]]
        true_pf_enc = [PUNCT_FINAL_LABELS.get(lbl, 0) for lbl in instance["punct_final"]]
        true_cap_enc = [int(c) for c in instance["capitalization"]]

        pred_pi_enc = [PUNCT_INITIAL_LABELS.get(lbl, 0) for lbl in predictions["punct_initial"]]
        pred_pf_enc = [PUNCT_FINAL_LABELS.get(lbl, 0) for lbl in predictions["punct_final"]]
        pred_cap_enc = [int(c) for c in predictions["capitalization"]]

        all_true_pi.extend(aggregate_word_level(instance["tokens"], true_pi_enc))
        all_pred_pi.extend(aggregate_word_level(predictions["tokens"], pred_pi_enc))

        all_true_pf.extend(aggregate_word_level(instance["tokens"], true_pf_enc))
        all_pred_pf.extend(aggregate_word_level(predictions["tokens"], pred_pf_enc))

        all_true_cap.extend(aggregate_word_level(instance["tokens"], true_cap_enc))
        all_pred_cap.extend(aggregate_word_level(predictions["tokens"], pred_cap_enc))

    pi_f1 = f1_score(all_true_pi, all_pred_pi, average="macro")
    pf_f1 = f1_score(all_true_pf, all_pred_pf, average="macro")
    cap_f1 = f1_score(all_true_cap, all_pred_cap, average="macro")

    print(f"Punctuation Initial F1: {pi_f1:.3f}")
    print(f"Punctuation Final F1: {pf_f1:.3f}")
    print(f"Capitalization F1: {cap_f1:.3f}")

    print("Initial punctuation classification report:")
    print(classification_report(
        all_true_pi, all_pred_pi,
        labels=list(PUNCT_INITIAL_LABELS.values()),
        target_names=list(PUNCT_INITIAL_LABELS.keys()),
        zero_division=0
    ))

    print("Final punctuation classification report:")
    print(classification_report(
        all_true_pf, all_pred_pf,
        labels=list(PUNCT_FINAL_LABELS.values()),
        target_names=list(PUNCT_FINAL_LABELS.keys()),
        zero_division=0
    ))

    print("Capitalization classification report:")
    print(classification_report(
        all_true_cap, all_pred_cap,
        labels=[0, 1, 2, 3],
        target_names=["lower", "Initial", "Mixed", "ALLCAP"],
        zero_division=0
    ))

    return pi_f1, pf_f1, cap_f1

def unify_subtoken_caps(tokens, caps):
    unified_caps = caps.copy()
    i = 0
    while i < len(tokens):
        if not tokens[i].startswith("##"):
            # Start of a new word
            start = i
            max_cap = caps[i]
            i += 1
            # Collect subtokens
            while i < len(tokens) and tokens[i].startswith("##"):
                if caps[i] > max_cap:
                    max_cap = caps[i]
                i += 1
            # Assign max_cap to all subtokens in the word
            for j in range(start, i):
                unified_caps[j] = max_cap
        else:
            # Subtoken without main token? Just skip
            i += 1
    return unified_caps
