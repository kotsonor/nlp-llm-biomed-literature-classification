import pytest
import pandas as pd
from pathlib import Path
import tempfile
from src.deployment.integrator import Integrator


class TestIntegrator:
    @pytest.fixture
    def sample_csv(self):
        # Create a temporary CSV for testing
        data = {
            "PMID": [12345, 67890],
            "Title": ["Test Title 1", "Test Title 2"],
            "Abstract": ["Test abstract 1", "Test abstract 2"],
        }
        df = pd.DataFrame(data)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            return f.name

    def test_init(self, sample_csv):
        integrator = Integrator(sample_csv)
        assert integrator.raw_path.exists()
        assert len(integrator.original_df) == 2
        assert integrator.pmid_col == "PMID"

    def test_reduce_columns(self, sample_csv):
        integrator = Integrator(sample_csv)
        integrator.reduce_columns(["PMID", "Title"])
        assert list(integrator.reduced_df.columns) == ["PMID", "Title"]
        assert len(integrator.reduced_df) == 2

    def test_merge_without_fetch(self, sample_csv):
        integrator = Integrator(sample_csv)
        integrator.reduce_columns(["PMID", "Title"])
        # Mock fetched_df
        integrator.fetched_df = pd.DataFrame({"PMID": [12345], "FetchedData": ["Data"]})
        integrator.merge()
        assert "FetchedData" in integrator.merged_df.columns
        assert len(integrator.merged_df) == 2  # Left join
