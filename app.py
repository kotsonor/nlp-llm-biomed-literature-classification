import streamlit as st
import pandas as pd
import os
from io import StringIO
import sys

# Append the 'src' folder to the system path to find the deployment module.
# This ensures that the script can locate and import your prediction function.
sys.path.append("src")

try:
    from deployment import predict_from_saved_model
except ImportError:
    st.error(
        "Could not find the 'deployment' module. Make sure app.py is in the main project folder "
        "and your prediction module is correctly placed in the 'src' folder."
    )
    st.stop()


class PubMedPredictorApp:
    """
    A Streamlit application for analyzing and predicting the relevance of PubMed articles
    based on user-provided PMIDs and keywords.
    """

    def __init__(self):
        """Initializes the application and sets the page configuration."""
        self.df = None
        self.uploaded_file = None
        st.set_page_config(
            page_title="Article-based Prediction", page_icon="🔬", layout="wide"
        )

    def _display_sidebar(self):
        """Displays the sidebar with instructions and warnings."""
        with st.sidebar:
            st.header("Instructions")
            st.info(
                """
                1.  **Upload a CSV file** containing PubMed identifiers.
                2.  **Verify the column name** containing the PMIDs.
                3.  **Enter keywords** separated by commas.
                4.  **Click 'Run Prediction'** to start the analysis.
                """
            )
            st.warning(
                "Ensure the `model` folder is in the same directory as this `app.py` file."
            )

    def _handle_file_upload(self):
        """Handles the CSV file upload and displays a preview of the data."""
        st.header("1. Upload Your Data File")
        self.uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if self.uploaded_file is not None:
            try:
                string_data = StringIO(self.uploaded_file.getvalue().decode("utf-8"))
                self.df = pd.read_csv(string_data)
                st.subheader("Uploaded Data Preview")
                st.dataframe(self.df.head())
            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")
                self.df = None

    def _get_pmid_column_name(self):
        """
        Gets the column name for PubMed IDs from the user and validates its existence.
        Returns the column name or None if invalid.
        """
        st.header("2. Specify PMID Column")

        pmid_column = st.text_input(
            "Enter the name of the column containing PubMed IDs (PMID)",
            value="PMID",
        )

        if self.df is not None:
            if pmid_column not in self.df.columns:
                st.error(
                    f"Error: Column '{pmid_column}' not found in the uploaded file. "
                    "Please check the column name and try again. "
                    f"Available columns: {', '.join(self.df.columns)}"
                )
                return None
        return pmid_column

    def _get_keywords(self):
        """Displays the text area for keyword input."""
        st.header("3. Provide Keywords")
        default_keywords = (
            "aggregates, amyloid, scfv, hiapp, mab, ttr, donanemab, aggregation"
        )
        return st.text_area(
            "Enter keywords separated by commas", value=default_keywords, height=100
        )

    def _run_prediction(self, keywords_input, pmid_column_name):
        """Handles the prediction logic when the user clicks the run button."""
        st.header("4. Run Analysis")
        if st.button("Run Prediction"):
            if self.df is not None and keywords_input and pmid_column_name:
                self._execute_model(keywords_input, pmid_column_name)
            else:
                st.warning(
                    "Please ensure a file is uploaded, a valid PMID column is specified, and keywords are provided."
                )

    def _execute_model(self, keywords_input, pmid_column_name):
        """Contains the core logic for running the model and displaying results."""
        try:
            keywords = [keyword.strip() for keyword in keywords_input.split(",")]
            st.info(f"Using keywords: {', '.join(keywords)}")
            st.info(f"Using column '{pmid_column_name}' for PubMed IDs.")

            with st.spinner(
                "Processing... The model is analyzing the data. This may take a moment."
            ):
                model_path = "model"
                temp_dataset_path = "temp_uploaded_data.csv"
                self.df.to_csv(temp_dataset_path, index=False)

                results_df = predict_from_saved_model(
                    model_path=model_path,
                    dataset_path=temp_dataset_path,
                    pmid_column_name=pmid_column_name,
                    keywords=keywords,
                    max_length=380,
                )

                os.remove(temp_dataset_path)

            self._display_results(results_df)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

    def _display_results(self, results_df):
        """Filters and displays the prediction results."""
        st.success("Prediction complete!")
        st.subheader("Results (articles marked as relevant - label: 1)")

        final_results = results_df[results_df["predicted_label"] == 1]

        if not final_results.empty:
            st.dataframe(final_results)
            csv = final_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="predicted_results.csv",
                mime="text/csv",
            )
        else:
            st.warning("No relevant articles were found for the given keywords.")

    def run(self):
        """The main method to run the Streamlit application."""
        st.title("🔬 PubMed Article Analysis and Prediction")
        st.write(
            "Upload a CSV file with a column of PubMed IDs and provide keywords to run the model."
        )

        self._display_sidebar()
        self._handle_file_upload()

        pmid_column_name = None

        if self.uploaded_file is not None:
            pmid_column_name = self._get_pmid_column_name()
            if pmid_column_name:
                keywords_input = self._get_keywords()
                self._run_prediction(keywords_input, pmid_column_name)


if __name__ == "__main__":
    app = PubMedPredictorApp()
    app.run()
