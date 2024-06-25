# Comparative Analysis of the Speeches of Barack Obama and Donald Trump

This project conducts a comparative analysis of speeches by Barack Obama and Donald Trump using NLP techniques. The objective is to identify and analyze linguistic patterns, emotional tones, and rhetorical styles that distinguish each speaker. The analysis uses sentiment analysis, POS tagging, NER, topic modeling, and semantic similarity. The goal is to visualize the results side by side in the end.

# To run the code you'll need:
 
 1. Python 3.x
 2. packages: json, re, spacy, nltk, nrclex, matplotlib, pandas, sklearn, sentence_transformers, torch


 There are two scripts: main.py; cosine_similarity.py

 # Order to run the scripts:
 1. main.py
 2. cosine_similarity.py

 # main.py


- normalize_text(text): Normalizes the text by converting symbols, removing times, capitalizing proper nouns and abbreviations, converting non-entity words to lowercase, and removing unwanted characters.
- process_json(file_path): Reads and normalizes the text data from a JSON file.
- save_normalized_data(data, output_file_path): Saves the normalized data to a JSON file.
- load_data(file_path): Loads data from a JSON file.
- process_text(data): Processes and categorizes text by POS with frequency count.
- save_data(data, file_path): Saves processed data to a JSON file.
- get_longest_sentences(data): Extracts and returns the 10 longest sentences by word count.
- get_catchphrases(data, min_n=3, max_n=None, top_n=10): Extracts the most common n-grams of variable length.
- visualize_catchphrases_side_by_side(trump_catchphrases, obama_catchphrases, output_file): Visualizes catchphrases in a side-by-side horizontal bar chart.
- analyze_emotions(text): Analyzes text for emotions and returns a dictionary of emotion percentages.
- visualize_emotions(file_path_trump, file_path_obama): Visualizes emotion scores by speaker.
- get_top_adjectives(data, top_n=10): Extracts the top adjectives by frequency.
- visualize_top_adjectives_side_by_side(file_path_trump, file_path_obama): Visualizes the top adjectives used by each speaker.
- split_into_chunks(text, chunk_size=100): Splits text into chunks of 100 words.
- display_topics(model, feature_names, no_top_words): Displays the top words for each topic.

# semantic_similarity.py


- read_json(file_path): Reads data from a JSON file.
- compute_similarity(text1, text2): Computes the cosine similarity between the embeddings of two texts from the normalized text in the json file.
