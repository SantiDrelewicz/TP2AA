import torch
import torch.nn as nn
import pandas as pd
from typing import List, Dict, Tuple, Optional

import torch.optim.lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertTokenizerFast
from sklearn.metrics import f1_score, classification_report

from utils import extract_labels

class PunctCapitalDataset(Dataset):
    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        return (
            torch.tensor(inst["input_ids"], dtype=torch.long),
            torch.tensor(inst["init_labels"], dtype=torch.long),
            torch.tensor(inst["final_labels"], dtype=torch.long),
            torch.tensor(inst["cap_labels"], dtype=torch.long),
        )


def collate_fn(batch):
    input_ids, init_labs, final_labs, cap_labs = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    init_labs = pad_sequence(init_labs, batch_first=True, padding_value=-100)
    final_labs = pad_sequence(final_labs, batch_first=True, padding_value=-100)
    cap_labs = pad_sequence(cap_labs, batch_first=True, padding_value=-100)
    return input_ids, init_labs, final_labs, cap_labs


class JointPunctCapitalModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_init: int,
        num_final: int,
        num_cap: int,
        n_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,     
    ):
        super().__init__()
        self.bidirectional = bidirectional

        lstm_hidden_size = hidden_dim // 2 if bidirectional else hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.output_dropout = nn.Dropout(dropout)
        self.init_head = nn.Linear(hidden_dim, num_init)
        self.final_head = nn.Linear(hidden_dim, num_final)
        self.cap_head = nn.Linear(hidden_dim, num_cap)


    def forward(self, x):
        # x: [B, T]
        emb = self.embedding(x)  # [B, T, E]
        emb = self.input_dropout(emb)  # dropout on embeddings

        out, _ = self.bilstm(emb)  # [B, T, H]
        out = self.output_dropout(out)  # dropout on LSTM outputs

        init_logits = self.init_head(out)  # [B, T, num_init]
        final_logits = self.final_head(out)  # [B, T, num_final]
        cap_logits = self.cap_head(out)  # [B, T, num_cap]
        return init_logits, final_logits, cap_logits

class RNNPunctuationCapitalizationModel:
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        lr_scheduler_patience: int = 2,
        early_stopping_patience: int = 3,
        batch_size: int = 128,
        bidirectional: bool = True,         
        device: Optional[torch.device] = None,
    ):
        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
        self.tokenizer_fast = BertTokenizerFast.from_pretrained("google-bert/bert-base-multilingual-cased")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.lr_scheduler_patience = lr_scheduler_patience
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model parameters
        self.num_init = 2
        self.num_final = 4
        self.num_cap = 4
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.is_fitted = False
        
        # Label mappings
        self.idx_map_init = {0: "", 1: "¿"}
        self.idx_map_final = {0: "", 1: ".", 2: "?", 3: ","}

    def _prepare_data(self, raw_sentences: List[str]) -> List[Dict]:
        """Process raw sentences and prepare training data"""
        
        data = []
        instances = []
        
        for inst_id, sentence in enumerate(raw_sentences, start=1):
            words, init_lbls, final_lbls, cap_lbls = extract_labels(sentence)
            token_idx = 0
            inst_data = []
            
            for word, init_lbl, final_lbl, cap_lbl in zip(
                words, init_lbls, final_lbls, cap_lbls
            ):
                subtokens = self.tokenizer.tokenize(word.lower())
                for i, sub in enumerate(subtokens):
                    # initial only on first subtoken
                    punct_init = init_lbl if i == 0 else ""
                    # final only on last subtoken
                    punct_final = final_lbl if i == len(subtokens) - 1 else ""
                    inst_data.append([inst_id, token_idx, sub, punct_init, punct_final, cap_lbl])
                    token_idx += 1
            
            if inst_id % 50_000 == 0:
                print(f"… processed {inst_id} sentences")
            
            data.extend(inst_data)
        
        # Convert to DataFrame and process
        df = pd.DataFrame(
            data,
            columns=[
                "inst_id",
                "token_id",
                "token",
                "punt_inicial",
                "punt_final",
                "capitalizacion",
            ],
        )
        
        # Convert token strings to BERT token IDs
        df["token_id_bert"] = self.tokenizer.convert_tokens_to_ids(df["token"].tolist())

        # Group by instance to form sequences
        grouped = {}
        for inst_id, group in df.groupby("inst_id"):
            grouped[inst_id] = {
                "input_ids": group["token_id_bert"].tolist(),
                "init_labels": [0 if lbl == "" else 1 for lbl in group["punt_inicial"]],
                "final_labels": [
                    0 if lbl == "" else (1 if lbl == "." else (2 if lbl == "?" else 3))
                    for lbl in group["punt_final"]
                ],
                "cap_labels": group["capitalizacion"].tolist(),
                "tokens": group["token"].tolist(),
            }

        return list(grouped.values())

    def _create_data_loaders(self, train_data: List[Dict], val_data: List[Dict]) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders"""
        train_loader = DataLoader(
            PunctCapitalDataset(train_data), 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            PunctCapitalDataset(val_data), 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        return train_loader, val_loader

    def train(self, training_data: List[str], validation_data: List[str], epochs: int = 3):
        """Train the RNN model"""
        print("Processing training data...")
        train_instances = self._prepare_data(training_data)
        print("Processing validation data...")
        val_instances = self._prepare_data(validation_data)
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(train_instances, val_instances)
        
        # Initialize model
        vocab_size = self.tokenizer.vocab_size
        self.model = JointPunctCapitalModel(
            vocab_size=vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_init=self.num_init,
            num_final=self.num_final,
            num_cap=self.num_cap,
            n_layers=self.n_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',         # Reduce LR when a monitored quantity has stopped decreasing
            factor=0.5,         # Factor by which the learning rate will be reduced. new_lr = lr * factor
            patience=self.lr_scheduler_patience, # Number of epochs with no improvement after which learning rate will be reduced.
        )

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Training on {self.device}")
        print(f"Train instances: {len(train_instances)}, Val: {len(val_instances)}")
        
        # Training loop
        scaler = GradScaler("cuda")

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            n_batches = 0
            
            for input_ids, init_labs, final_labs, cap_labs in train_loader:
                input_ids = input_ids.to(self.device)
                init_labs = init_labs.to(self.device)
                final_labs = final_labs.to(self.device)
                cap_labs = cap_labs.to(self.device)

                self.optimizer.zero_grad()

                with autocast("cuda"):
                    init_logits, final_logits, cap_logits = self.model(input_ids)

                    loss_init = self.criterion(init_logits.view(-1, self.num_init), init_labs.view(-1))
                    loss_final = self.criterion(final_logits.view(-1, self.num_final), final_labs.view(-1))
                    loss_cap = self.criterion(cap_logits.view(-1, self.num_cap), cap_labs.view(-1))
                    loss = loss_init + loss_final + loss_cap

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

                running_loss += loss.item()
                n_batches += 1

            avg_train_loss = running_loss / n_batches
            print(f"Epoch {epoch} — Train loss: {avg_train_loss:.4f}")

            avg_val_loss = self._validate_and_return_loss(val_loader, epoch) # Call the new helper method
            self.scheduler.step(avg_val_loss)

            print("-" * 60)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break
            
        self.is_fitted = True
        print("Training completed!")

    def _validate_and_return_loss(self, val_loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        val_loss = 0.0
        n_val_batches = 0

        all_init_trues, all_init_preds = [], []
        all_final_trues, all_final_preds = [], []
        all_cap_trues, all_cap_preds = [], []

        with torch.no_grad():
            for input_ids, init_labs, final_labs, cap_labs in val_loader:
                input_ids = input_ids.to(self.device)
                init_labs = init_labs.to(self.device)
                final_labs = final_labs.to(self.device)
                cap_labs = cap_labs.to(self.device)

                init_logits, final_logits, cap_logits = self.model(input_ids)

                loss_init = self.criterion(init_logits.view(-1, self.num_init), init_labs.view(-1))
                loss_final = self.criterion(final_logits.view(-1, self.num_final), final_labs.view(-1))
                loss_cap = self.criterion(cap_logits.view(-1, self.num_cap), cap_labs.view(-1))
                loss = loss_init + loss_final + loss_cap
                val_loss += loss.item()
                n_val_batches += 1

                init_preds = init_logits.argmax(dim=-1)
                final_preds = final_logits.argmax(dim=-1)
                cap_preds = cap_logits.argmax(dim=-1)

                mask_init = init_labs.view(-1) != -100
                mask_final = final_labs.view(-1) != -100
                mask_cap = cap_labs.view(-1) != -100

                all_init_trues.extend(init_labs.view(-1)[mask_init].cpu().tolist())
                all_init_preds.extend(init_preds.view(-1)[mask_init].cpu().tolist())
                all_final_trues.extend(final_labs.view(-1)[mask_final].cpu().tolist())
                all_final_preds.extend(final_preds.view(-1)[mask_final].cpu().tolist())
                all_cap_trues.extend(cap_labs.view(-1)[mask_cap].cpu().tolist())
                all_cap_preds.extend(cap_preds.view(-1)[mask_cap].cpu().tolist())

        avg_val_loss = val_loss / n_val_batches
        print(f"Epoch {epoch} — Val loss:   {avg_val_loss:.4f}")

        f1_init_macro = f1_score(all_init_trues, all_init_preds, average="macro", zero_division=0)
        f1_final_macro = f1_score(all_final_trues, all_final_preds, average="macro", zero_division=0)
        f1_cap_macro = f1_score(all_cap_trues, all_cap_preds, average="macro", zero_division=0)
        print(f"Epoch {epoch} — F1 (macro): init={f1_init_macro:.3f}, final={f1_final_macro:.3f}, cap={f1_cap_macro:.3f}")

        print("\nInitial punctuation per-class F1:")
        print(classification_report(
            all_init_trues, all_init_preds,
            labels=[0, 1], target_names=["no-¿", "¿"], zero_division=0,
        ))

        print("Final punctuation per-class F1:")
        print(classification_report(
            all_final_trues, all_final_preds,
            labels=[0, 1, 2, 3], target_names=["none", ".", "?", ","], zero_division=0,
        ))

        print("Capitalization per-class F1:")
        print(classification_report(
            all_cap_trues, all_cap_preds,
            labels=[0, 1, 2, 3], target_names=["lower", "Initial", "Mixed", "ALLCAP"], zero_division=0,
        ))
        
        return avg_val_loss

    def predict(self, text: str):
        """Predict punctuation and capitalization for input text"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        return self.predict_and_reconstruct(text)

    def predict_and_fill_csv(self, input_df: pd.DataFrame, output_file: str = "predicted.csv") -> pd.DataFrame:
        """
        Takes a dataframe with columns: instancia_id, token_id, token
        Returns a new dataframe with added columns:
        punt_inicial, punt_final, capitalización
        One row per *input token* (same granularity).
        """
        import pandas as pd
        import torch

        results = []
        tokenizer_fast = self.tokenizer_fast
        device = self.device

        for instancia_id, group in input_df.groupby("instancia_id"):
            # 1. Get the *input tokens exactly as they appear* (these are already subword tokens)
            tokens = group["token"].tolist()

            # 2. Convert tokens to IDs using tokenizer's vocab
            input_ids = tokenizer_fast.convert_tokens_to_ids(tokens)
            input_ids_tensor = torch.tensor([input_ids], device=device)

            # 3. Predict
            self.model.eval()
            with torch.no_grad():
                init_logits, final_logits, cap_logits = self.model(input_ids_tensor)

            # 4. Get predictions per token
            init_pred = init_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
            final_pred = final_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
            cap_pred = cap_logits.argmax(dim=-1).squeeze(0).cpu().tolist()

            # 5. Decode label indices
            punt_inicial = [self.idx_map_init[idx] for idx in init_pred]
            punt_final = [self.idx_map_final[idx] for idx in final_pred]
            capitalizacion = cap_pred  # leave as integers or map if you want

            # 6. Build output dataframe for this group
            predicted_group = group.copy()
            predicted_group["punt_inicial"] = punt_inicial
            predicted_group["punt_final"] = punt_final
            predicted_group["capitalización"] = capitalizacion

            results.append(predicted_group)

        # Concatenate all
        final_df = pd.concat(results, ignore_index=True)

        if output_file:
            final_df.to_csv(output_file, index=False)

        return final_df

    def predict_and_reconstruct(self, raw_sentence: str) -> str:
        """
        Runs the model on a single raw sentence and returns the
        reconstructed sentence with punctuation & capitalization.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        tokenizer_fast = self.tokenizer_fast
        
        # 1) Tokenize (with word-ids for alignment)
        enc = tokenizer_fast(
            raw_sentence.lower().split(),  # split into words so word_ids works
            is_split_into_words=True,
            return_offsets_mapping=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        input_ids = enc["input_ids"].to(self.device)
        word_ids = enc.word_ids(batch_index=0)  # list of length L

        # 2) Model forward
        self.model.eval()
        with torch.no_grad():
            init_logits, final_logits, cap_logits = self.model(input_ids)
        init_pred = init_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        final_pred = final_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        cap_pred = cap_logits.argmax(dim=-1).squeeze(0).cpu().tolist()

        # 3) Gather per-word predictions
        words: List[str] = []
        cur_word_idx = None
        cur_subtokens: List[str] = []
        cur_init = ""
        cur_cap = 0
        cur_final = ""

        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            token = tokenizer_fast.convert_ids_to_tokens(int(input_ids[0, i]))
            # start of a new word?
            if wid != cur_word_idx:
                # flush previous
                if cur_word_idx is not None:
                    # assemble the word text
                    word_text = "".join(cur_subtokens)
                    # apply capitalization
                    if cur_cap == 3:
                        word_text = word_text.upper()
                    elif cur_cap == 1:
                        word_text = word_text.capitalize()
                    elif cur_cap == 2:
                        if len(word_text) > 1:
                            word_text = word_text[0].upper() + word_text[1:]
                        else:
                            word_text = word_text.upper()
                    # attach final punctuation
                    word_text = word_text + self.idx_map_final[cur_final]
                    # prepend initial punctuation if any
                    word_text = cur_init + word_text
                    words.append(word_text)
                # reset for new word
                cur_word_idx = wid
                cur_subtokens = [token.replace("##", "")]  # start fresh
                cur_init = self.idx_map_init[init_pred[i]]
                cur_final = final_pred[i]
                cur_cap = cap_pred[i]
            else:
                # continuing same word
                cur_subtokens.append(token.replace("##", ""))
                # update final & cap to last sub-token's prediction
                cur_final = final_pred[i]
                # we keep init and cap from first subtoken
        # flush last word
        if cur_word_idx is not None:
            word_text = "".join(cur_subtokens)
            if cur_cap == 3:
                word_text = word_text.upper()
            elif cur_cap == 1:
                word_text = word_text.capitalize()
            elif cur_cap == 2:
                word_text = word_text[0].upper() + word_text[1:]
            word_text = word_text + self.idx_map_final[cur_final]
            word_text = cur_init + word_text
            words.append(word_text)

        # finally, join with spaces:
        return " ".join(words)

    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
                "num_init": self.num_init,
                "num_final": self.num_final,
                "num_cap": self.num_cap,
                "vocab_size": self.tokenizer.vocab_size,
                "bidirectional": self.bidirectional,
            },
            "training_config": {
                "learning_rate": self.learning_rate,
                "lr_scheduler_patience": self.lr_scheduler_patience,
                "early_stopping_patience": self.early_stopping_patience,
                "batch_size": self.batch_size,
            },
            "is_fitted": self.is_fitted,
        }

        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = torch.load(filepath, map_location=self.device)
        
        # Update configs
        config = model_data["model_config"]
        self.embed_dim = config["embed_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.n_layers = config["n_layers"]
        self.dropout = config["dropout"]
        self.num_init = config["num_init"]
        self.num_final = config["num_final"]
        self.num_cap = config["num_cap"]
        self.bidirectional = config["bidirectional"]
        
        train_config = model_data["training_config"]
        self.learning_rate = train_config["learning_rate"]
        self.batch_size = train_config["batch_size"]
        
        # Recreate model
        self.model = JointPunctCapitalModel(
            vocab_size=config["vocab_size"],
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_init=self.num_init,
            num_final=self.num_final,
            num_cap=self.num_cap,
            n_layers=self.n_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)
        
        # Load state
        self.model.load_state_dict(model_data["model_state_dict"])
        self.is_fitted = model_data["is_fitted"]
        
        print(f"Model loaded from {filepath}")

def evaluate_model_rnn(rnn: RNNPunctuationCapitalizationModel, test_sentences: List[str], batch_size: Optional[int] = None):
    """
    Evaluate the RNNPunctuationCapitalizationModel on test data with labels.
    Prints classification reports and F1 scores for each task.
    
    Args:
        model: Trained RNNPunctuationCapitalizationModel instance.
        test_sentences: List of raw test sentences (strings).
        batch_size: Optional batch size for evaluation; defaults to model's batch_size.
    """
    if not rnn.is_fitted:
        raise ValueError("Model must be trained before evaluation")
    
    rnn.model.eval()
    device = rnn.device
    bs = batch_size or rnn.batch_size

    # Prepare data instances and dataloader
    test_instances = rnn._prepare_data(test_sentences)
    test_loader = DataLoader(
        PunctCapitalDataset(test_instances),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_init_trues, all_init_preds = [], []
    all_final_trues, all_final_preds = [], []
    all_cap_trues, all_cap_preds = [], []

    with torch.no_grad():
        for input_ids, init_labs, final_labs, cap_labs in test_loader:
            input_ids = input_ids.to(device)
            init_labs = init_labs.to(device)
            final_labs = final_labs.to(device)
            cap_labs = cap_labs.to(device)

            init_logits, final_logits, cap_logits = rnn.model(input_ids)

            init_pred = init_logits.argmax(dim=-1)
            final_pred = final_logits.argmax(dim=-1)
            cap_pred = cap_logits.argmax(dim=-1)

            # Mask padding tokens (-100)
            mask_init = init_labs.view(-1) != -100
            mask_final = final_labs.view(-1) != -100
            mask_cap = cap_labs.view(-1) != -100

            all_init_trues.extend(init_labs.view(-1)[mask_init].cpu().tolist())
            all_init_preds.extend(init_pred.view(-1)[mask_init].cpu().tolist())

            all_final_trues.extend(final_labs.view(-1)[mask_final].cpu().tolist())
            all_final_preds.extend(final_pred.view(-1)[mask_final].cpu().tolist())

            all_cap_trues.extend(cap_labs.view(-1)[mask_cap].cpu().tolist())
            all_cap_preds.extend(cap_pred.view(-1)[mask_cap].cpu().tolist())

    # Compute and print overall macro F1 scores
    f1_init = f1_score(all_init_trues, all_init_preds, average="macro", zero_division=0)
    f1_final = f1_score(all_final_trues, all_final_preds, average="macro", zero_division=0)
    f1_cap = f1_score(all_cap_trues, all_cap_preds, average="macro", zero_division=0)
    print(f"Test set macro F1 scores:")
    print(f"  Initial punctuation: {f1_init:.4f}")
    print(f"  Final punctuation:   {f1_final:.4f}")
    print(f"  Capitalization:      {f1_cap:.4f}\n")

    # Print detailed classification reports
    print("Initial punctuation classification report:")
    print(classification_report(
        all_init_trues, all_init_preds,
        labels=[0, 1], target_names=["no-¿", "¿"], zero_division=0
    ))

    print("Final punctuation classification report:")
    print(classification_report(
        all_final_trues, all_final_preds,
        labels=[0, 1, 2, 3], target_names=["none", ".", "?", ","], zero_division=0
    ))

    print("Capitalization classification report:")
    print(classification_report(
        all_cap_trues, all_cap_preds,
        labels=[0, 1, 2, 3], target_names=["lower", "Initial", "Mixed", "ALLCAP"], zero_division=0
    ))