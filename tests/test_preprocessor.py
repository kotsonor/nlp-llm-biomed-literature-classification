import pytest
import pandas as pd
from src.deployment.preprocessor import Preprocessor


class TestPreprocessor:
    @pytest.fixture
    def sample_df(self):
        data = {
            "PMID": [12345, 67890, None],
            "Abstract": ["Abstract 1", None, "Abstract 3"],
            "Label": ["Useful", "Rejected", "Useful"],
        }
        return pd.DataFrame(data)

    def test_init(self, sample_df):
        prep = Preprocessor(sample_df)
        assert len(prep.df) == 3
        assert prep.dropped_df is None

    def test_dropna(self, sample_df):
        prep = Preprocessor(sample_df)
        prep.dropna(subset=["Abstract"])
        assert len(prep.df) == 2  # One row with NaN in Abstract
        assert len(prep.dropped_df) == 1

    def test_map_labels(self, sample_df):
        prep = Preprocessor(sample_df)
        mapping = {"Rejected": 0, "Useful": 1}
        prep.map_labels("Label", mapping)
        assert prep.df["Label"].tolist() == [1, 0, 1]
