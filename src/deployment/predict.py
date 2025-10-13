import pandas as pd
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.dataset_builder import DataFrameConverter
from src.deployment.preprocessor import Preprocessor
from src.deployment.integrator import Integrator


def predict_from_saved_model(
    model_path: str,
    dataset_path: str,
    pmid_column_name: str,
    keywords: list[str],
    max_length: int,
) -> pd.DataFrame:
    """
    Loads a trained BERT sequence classification model from disk and performs
    predictions on a new dataset, fetching abstracts from PubMed first.

    Args:
        model_path (str): The path to the directory containing the saved model and tokenizer (e.g., 'bert-base-uncased' or a local path).
        dataset_path (str): The path to the input file (e.g., CSV) containing the identifiers (PMIDs) for which to get predictions.
        pmid_column_name (str): The name of the column in the input file that contains the article identifiers (e.g., "PMID").
        keywords (list[str]): A list of keywords used during tokenization and feature construction by the 'DataFrameConverter'.
        max_length (int): The maximum sequence length for the BERT model tokenizer.

    Returns:
        pd.DataFrame: A DataFrame containing the article identifiers (PMIDs) and their predicted labels (0 or 1).
    """

    integrator = Integrator(
        raw_path=dataset_path, pmid_col=pmid_column_name, email="test@gmail.com"
    )
    integrator.reduce_columns(keep_columns=[pmid_column_name])
    integrator.fetch_pubmed()
    integrator.merge()

    merged = integrator.merged_df.copy()
    preprocessor = Preprocessor(merged)

    preprocessor.dropna(subset=["Abstract"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)  # Move the model to the appropriate device

    # Create test dataset
    test_df = preprocessor.df.copy()
    df_converter = DataFrameConverter()
    test_dataset = df_converter.convert(test_df, tokenizer, keywords, max_length)
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    training_args = TrainingArguments(
        output_dir="./results", report_to="none", eval_strategy="no"
    )
    eval_trainer = Trainer(model=model, args=training_args)

    # Perform predictions
    predictions_output = eval_trainer.predict(test_dataset["test"])
    logits = predictions_output.predictions
    predicted_labels = np.argmax(logits, axis=1)

    logits_tensor = torch.from_numpy(logits)
    probabilities = torch.nn.functional.softmax(logits_tensor, dim=1)
    probability_of_class_1 = probabilities[:, 1].numpy()

    test_dataset.reset_format()
    pmids = test_dataset["test"][pmid_column_name]
    results_df = pd.DataFrame(
        {
            pmid_column_name: pmids,
            "predicted_label": predicted_labels,
            "probability": probability_of_class_1,
        }
    )

    print("Prediction complete!")
    return results_df
