# Extracting Book Titles from Historical Newspapers

This repository contains the code and resources for my thesis project titled “Extracting Book Titles from Historical Newspaper Archives: A Named Entity Recognition Approach.” The project aims to autonomously extract book titles from OCR-scanned historical newspapers using advanced Named Entity Recognition (NER) techniques.

## Thesis Summary

In this project, I developed a novel method for identifying book titles in Dutch historical newspaper archives, specifically focusing on the Leeuwarder Courant, Het parool and Trouw. The project utilizes various NER models, including BiLSTM-CRF and transformer-based models like XLM-RoBERTa. The transformer models achieved the best performance, with an F1 score of 84.3% on the test dataset.

## Model

The final model for Named Entity Recognition (NER) of book titles can be found on [Hugging Face](https://huggingface.co/Nielsaxe/BookTitleNERDutch).
## Repository Structure

- `notebooks/`: Jupyter notebooks for data analysis and experimentation.
- `scripts/`: Python scripts for data preparation, model training, evaluation, and visualization.
- `thesis/`: Documents related to the thesis.
- `word_dictionary/`: Dictionary of words for OCR analysis, including Dutch and German dictionaries used to assess OCR quality.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: Project overview and setup instructions.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nielsbijl/BookReviewsThesis.git
   cd BookReviewsThesis

