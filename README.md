# Project Description

This project focuses on the **automatic classification of biomedical texts** to assist researchers in conducting efficient literature reviews. The main goal is to identify scientific articles describing the **experimental effects of antibodies on amyloid formation**.

## Model Development Process

The project involved several key stages:

1. **Manual Dataset Creation**
   A custom dataset was built by manually selecting and evaluating articles from the PubMed database, ensuring high-quality labels.

2. **Hyperparameter Tuning**
   A series of experiments were conducted to identify the optimal hyperparameters. The tested settings included:
   * Tokenizer's `max_length`.
   * Learning rate (`lr`).
   * Number of frozen vs. trainable layers to balance performance and training time.
   * Every experiment was logged on [Weights & Biases](https://api.wandb.ai/links/axzions-university-of-wroclaw/fyqsf6bc).

3. **Model Training**
   After tuning, the final model was trained on the prepared dataset.

4. **Deployment and Sharing**

   * An interactive **Streamlit** application allows easy predictions on new data: [Deep Skim App](https://kotsonor-deep-skim-app-6wrb4z.streamlit.app/).
   * The trained model is available on the **Hugging Face Hub** under `kotsonor/biomedBERT-Amyloid`: [Hugging Face Model](https://huggingface.co/kotsonor/BiomedBERT-Amyloid).



## Project Structure

```
app.py                  # Streamlit application for predictions
Dockerfile              # Docker configuration for containerized deployment
docker-compose.yml      # Docker Compose for easy setup
pytest.ini             # Pytest configuration
requirements.txt        # Python dependencies
data/                   # Folder for raw and processed datasets 
model/                  # Directory to store trained models
notebooks/              # Jupyter notebooks for experiments and analysis
src/                    # Source code for data processing and model training
  ├─ deployment/        # Scripts and utilities for deploying the model
  ├─ fetchers/          # Scripts to fetch data from PubMed or other sources
  ├─ data_prep.py       # Functions for preprocessing and cleaning data
  ├─ data_splitter.py   # Code to split datasets into train/validation/test sets
  ├─ dataset_builder.py # Classes to convert pandas df into tokenized HuggingFace Datasets  
  ├─ fetch_data.py      # CLI script to fetch publication data 
  ├─ model_building.py  # Implements the model training pipeline using a Strategy pattern
  └─ run.py             # Main script to run training pipelines, hyperparameter search
tests/                  # Unit tests for the application
```

## Installation and Usage

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kotsonor/nlp-llm-biomed-literature-classification.git
   cd nlp-llm-biomed-literature-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (create `.env` file):
   ```
   EMAIL=your_email@example.com
   API_WANDB=your_wandb_api_key
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Docker Installation
1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```
   The app will be available at `http://localhost:8501`.

   **Note**: The model is downloaded automatically from Hugging Face on first run. Ensure you have a stable internet connection. For offline use, download the model locally to the `model/` directory before running.

### Running Tests
To run the test suite:
```bash
pytest
```

### CI/CD
The project uses GitHub Actions for continuous integration. Tests are automatically run on every push and pull request.

## Dataset

The dataset was created from scratch specifically for this project.
* **Source**: PubMed database.
* **Process**: Articles were retrieved using specific search queries, and each one was then **manually assessed** and classified as "useful" (describing the experimental effect of antibodies on amyloids) or "not useful".
* **Search Queries Used**:
    * `"amyloid"[Title/Abstract] AND "antibod*"[Title/Abstract]`
    * `"amyloid"[Title/Abstract] AND "nanobod*"[Title/Abstract]`
* **Statistics**:
    * **Total Articles**: 1939
    * **Useful Articles (label 1)**: 167 (9%)
    * **Not Useful Articles (label 0)**: 1772 (91%)


## Features
* **Automation**: Scripts for automatically fetching article metadata from PubMed.
* **End-to-End Pipeline**: Covers the full project lifecycle, from raw data collection to model deployment.
* **Flexibility**: Allows configuration of key hyperparameters like `max_length`, `learning rate`, the number of trainable layers, and many more. 
* **Interactive Application**: A Streamlit web app enables easy model testing and prediction without requiring coding knowledge.
* **Open Access**: The model is publicly available on the Hugging Face Hub, ready for use in other projects.


## Acknowledgments

We gratefully acknowledge the support for this research from:
- **Institution:** Bioinformatics and Multiomics Analysis Laboratory, Clinical Research Centre, Medical University of Bialystok.
- **Funding Source:** National Science Center, Poland, via the SONATA 19 grant.
- **Project No:** DEC-2023/51/D/NZ7/02847.
- **Project Title:** “Taming aggregation with AmyloGraphem 2.0: database and predictive model of amyloid self-organization of modulators”.

We also thank Valentín Iglesias and Mariia Solovianova from the Medical University of Białystok for preparing the dataset. 