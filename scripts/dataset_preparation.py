import os
import pickle
import string
import numpy as np
from tqdm import tqdm
import re


def remove_extra_spaces(text):
    """
    Remove extra spaces from a string by replacing multiple spaces with a single space.

    Args:
        text (str): The input text with potential extra spaces.

    Returns:
        str: The cleaned text with single spaces between words.
    """
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text.strip()


def remove_punctuation(input_string):
    """
    Remove punctuation from a given string, including old Dutch quotation marks.

    Args:
        input_string (str): The input string from which punctuation will be removed.

    Returns:
        str: The input string without punctuation.
    """
    extended_punctuation = string.punctuation + '„“'
    translator = str.maketrans('', '', extended_punctuation)
    return input_string.translate(translator)


def find_sentence_in_text(full_text, sentence):
    """
    Find the start and end indices of a sentence within a full text.

    Args:
        full_text (str): The text in which to search for the sentence.
        sentence (str): The sentence to find within the full text.

    Returns:
        tuple: The start and end indices of the sentence within the full text.

    Raises:
        ValueError: If the sentence is not found in the full text.
    """
    start_index = full_text.find(sentence)
    if start_index == -1:
        raise ValueError("Sentence not found in text.")
    end_index = start_index + len(sentence)
    return start_index, end_index


def create_mask_for_sentence(doc, start_index, end_index):
    """
    Create a binary mask for a sentence within a tokenized document.

    Args:
        doc (spacy.tokens.doc.Doc): The tokenized document.
        start_index (int): The start index of the sentence in the document.
        end_index (int): The end index of the sentence in the document.

    Returns:
        list: A binary mask indicating the tokens that are part of the sentence.
    """
    mask = [0] * len(doc)
    for i, token in enumerate(doc):
        token_end_idx = token.idx + len(token.text)
        if token.idx <= end_index and token_end_idx >= start_index:
            mask[i] = 1
    return mask


def process_text(text, nlp, remove_punc=False, force_lower_case=False):
    """
    Process text by optionally removing punctuation and converting to lowercase.

    Args:
        text (str): The input text to process.
        nlp (spacy.lang.xx.XX): The spaCy language model.
        remove_punc (bool): Whether to remove punctuation from the text.
        force_lower_case (bool): Whether to convert the text to lowercase.

    Returns:
        tuple: The tokenized document and list of tokens.
    """
    if remove_punc:
        text = remove_punctuation(text)
    doc = nlp(text)
    tokens = [token.text.lower() if force_lower_case else token.text for token in doc]
    return doc, tokens


def create_data_set(samples, df, nlp, remove_punc=False, force_lower_case=False):
    """
    Create a dataset for named entity recognition (NER) by generating masks for book titles in text samples.

    Args:
        samples (list): The list of text samples.
        df (pandas.DataFrame): The dataframe containing the text samples and book titles.
        nlp (spacy.lang.xx.XX): The spaCy language model.
        remove_punc (bool): Whether to remove punctuation from the text samples.
        force_lower_case (bool): Whether to convert the text samples to lowercase.

    Returns:
        list: A list of dictionaries containing tokens and NER tags.
    """
    data = []
    for sample in tqdm(samples):
        unique_content_df = df[df['content'] == sample]
        doc, tokens = process_text(sample, nlp, remove_punc, force_lower_case)
        masks = []

        for _, row in unique_content_df.iterrows():
            try:
                start_index, end_index = find_sentence_in_text(sample.lower(), row['title4'].lower())
                mask = create_mask_for_sentence(doc, start_index, end_index)
                masks.append(mask)
            except ValueError:
                continue

        if masks:
            combined_mask = np.bitwise_or.reduce(np.array(masks), axis=0)
            data.append({"tokens": tokens, "ner_tags": combined_mask})
    return data


def trouw_parool_create_dataset(df, nlp, remove_punc=False, force_lower_case=False):
    """
    Create a dataset for named entity recognition (NER) from "Parool" and "Trouw" newspapers.

    Args:
        df (pandas.DataFrame): The dataframe containing the text samples and indices for book titles.
        nlp (spacy.lang.xx.XX): The spaCy language model.
        remove_punc (bool): Whether to remove punctuation from the text samples.
        force_lower_case (bool): Whether to convert the text samples to lowercase.

    Returns:
        list: A list of dictionaries containing tokens and NER tags.
    """
    data = []
    for sample in tqdm(df['text'].unique()):
        unique_content_df = df[df['text'] == sample]
        doc, tokens = process_text(sample, nlp, remove_punc, force_lower_case)
        masks = []

        for _, row in unique_content_df.iterrows():
            mask = create_mask_for_sentence(doc, row['start_index'], row['end_index'])
            masks.append(mask)

        if masks:
            combined_mask = np.bitwise_or.reduce(np.array(masks), axis=0)
            data.append({"tokens": tokens, "ner_tags": combined_mask})
    return data


def save_dataset(dataset, filename):
    """
    Save a dataset to a file using pickle.

    Args:
        dataset (list): The dataset to save.
        filename (str): The file path where the dataset will be saved.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


def load_dataset(filename):
    """
    Load a dataset from a file using pickle.

    Args:
        filename (str): The file path from where the dataset will be loaded.

    Returns:
        list: The loaded dataset.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def split_samples(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split samples into training, validation, and test sets.

    Args:
        samples (list): The list of samples to split.
        train_ratio (float): The ratio of samples for the training set.
        val_ratio (float): The ratio of samples for the validation set.
        test_ratio (float): The ratio of samples for the test set.

    Returns:
        tuple: The training, validation, and test sets.
    """
    np.random.shuffle(samples)
    train_end = int(len(samples) * train_ratio)
    val_end = train_end + int(len(samples) * val_ratio)
    return samples[:train_end], samples[train_end:val_end], samples[val_end:]
