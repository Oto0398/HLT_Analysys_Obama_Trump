import json
import re
import spacy
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk import ngrams,sent_tokenize, word_tokenize
from collections import Counter
from nrclex import NRCLex
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # Load English stopwords
nlp = spacy.load("en_core_web_md")
###

def normalize_text(text): # normalization
    text = text.replace(u'\u2019', "'").replace(u'\u2018', "'")  # Convert curly apostrophes to straight ones
    text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text) # Remove times like 12:30
    doc = nlp(text) # Apply spaCy NER to capitalize proper nouns and abbreviations
    tokens = []
    for token in doc:
        if token.ent_type_:
            tokens.append(token.text_with_ws) # Preserve and capitalize entities as they are detected
        else:
            if token.text.isupper() and len(token.text) > 1: # Handle non-entities
                tokens.append(token.text_with_ws)  # Preserving uppercase abbreviations like USA, U.S., etc.
            else:
                # Apply lowercase to non-entity, non-abbreviation words
                tokens.append(token.text_with_ws.lower())  # Apply lowercase to non-entity, non-abbreviation words
    
    normalized_text = ''.join(tokens)

    normalized_text = re.sub(r'[^\w\s,.\'-]', '', normalized_text) # Remove unwanted characters, but keep periods, commas, and apostrophes

    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip() # Replace multiple spaces with a single space and strip leading/trailing spaces

    return normalized_text

def process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for entry in data:
        entry['text'] = normalize_text(entry['text'])
    return data
#
def save_normalized_data(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

json_files = ['trump_speeches.json', 'obama_speeches.json'] # List of files to be processed

for file_path in json_files:
    normalized_data = process_json(file_path)
    output_file_path = 'normalized_' + file_path
    save_normalized_data(normalized_data, output_file_path)
    print(f"Normalized data saved to {output_file_path}")

def load_data(file_path):  #Load data from a JSON file """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def process_text(data):  #Replace the old function with this one to process and categorize text by POS with frequency count"""
    exclude_words = {'barack', 'obama', 'president', 'donald', 'trump'}
    pos_categories = defaultdict(Counter)
    for entry in data:
        doc = nlp(entry['text'])
        for token in doc:
            if token.text.lower() in exclude_words or token.pos_ in ['PUNCT', 'SPACE', 'SYM']:
                continue
            pos_categories[token.pos_].update([token.text.lower()])
    sorted_pos_categories = {pos: sorted(words.items(), key=lambda x: -x[1]) for pos, words in pos_categories.items()}
    return sorted_pos_categories

def save_data(data, file_path): #Save processed data to a JSON file, handling different types of structures."""
    if isinstance(data, dict) and all(isinstance(val, dict) for val in data.values()):
        readable_data = {pos: {word: freq for word, freq in words.items()} for pos, words in data.items()} # This handles dictionaries of dictionaries (as used for POS counts)
    else:
        readable_data = data         # This handles lists (as used for longest sentences)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(readable_data, file, indent=4)

def get_longest_sentences(data):
    sentences = [] #Extract and return the 10 longest sentences from the provided data by word count.
    for entry in data:
        doc = nlp(entry['text'])
        sentences.extend([(sent.text.strip(), len(sent.text.split())) for sent in doc.sents])  # Here, we count the number of words in each sentence using len(sent.text.split())

    longest_sentences = sorted(sentences, key=lambda x: x[1], reverse=True)[:10]     # Sort sentences by word count in descending order and return the top 10
    
    return longest_sentences # Return a list of tuples, each containing the sentence text and its word count

def get_catchphrases(data, min_n=3, max_n=None, top_n=10):
    """Extract the most common n-grams of variable length from provided text data."""
    exclude_words = {'barack', 'obama', 'donald', 'trump'}  # Words to exclude
    exclude_chars = {'"', "'", ".", ",", ":", ";", "!", "?"}  # Punctuation to exclude
    phrase_counter = Counter()

    for text in data:
        sentences = sent_tokenize(text.lower())  # Tokenize into sentences

        for sentence in sentences:
            sentence = ''.join(ch for ch in sentence if ch not in exclude_chars)  # Remove punctuation
            words = word_tokenize(sentence)
            words = [word for word in words if word not in exclude_words and word.isalpha()]  # Filter out excluded words

            # Determine the max length of n-grams for this sentence
            sentence_max_n = max_n if max_n is not None else len(words)
            for n in range(min_n, sentence_max_n + 1):
                for ngram in ngrams(words, n):
                    phrase_counter[ngram] += 1

    # Convert n-grams from tuples to strings and sort by frequency
    top_phrases = {' '.join(phrase): count for phrase, count in phrase_counter.most_common(top_n)}
    return top_phrases

def visualize_catchphrases_side_by_side(trump_catchphrases, obama_catchphrases, output_file):
    """Visualize the catchphrases in a side by side horizontal bar chart."""
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Trump's catchphrases on the left
    if trump_catchphrases:
        trump_phrases, trump_counts = zip(*sorted(trump_catchphrases.items(), key=lambda x: x[1], reverse=True))
        axs[0].barh(trump_phrases, trump_counts, color='tomato')
    axs[0].set_title('Most Frequently Used Catchphrases by Trump')

    # Obama's catchphrases on the right
    if obama_catchphrases:
        obama_phrases, obama_counts = zip(*sorted(obama_catchphrases.items(), key=lambda x: x[1], reverse=True))
        axs[1].barh(obama_phrases, obama_counts, color='dodgerblue')
    axs[1].set_title('Most Frequently Used Catchphrases by Obama')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

#
def analyze_emotions(text):  #Analyze text for emotions and return a dictionary of emotion percentages."""
    emotion_analyzer = NRCLex(text)
    emotion_frequencies = emotion_analyzer.affect_frequencies
    
    # Map NRCLex emotions to your specified categories
    mapped_emotions = {
        'Anger': emotion_frequencies.get('anger', 0),
        'Fear': emotion_frequencies.get('fear', 0),
        'Disgust': emotion_frequencies.get('disgust', 0),
        'Pain': emotion_frequencies.get('negative', 0),  # Assuming 'negative' as a proxy for pain
        'Sad': emotion_frequencies.get('sadness', 0),
        'Happy': emotion_frequencies.get('joy', 0),
        'Hopeful': emotion_frequencies.get('anticipation', 0),  # Assuming 'anticipation' as a proxy for hopefulness
        'Gratitude': emotion_frequencies.get('trust', 0)  # Assuming 'trust' as a proxy for gratitude
    }
    
    total = sum(mapped_emotions.values())  # Normalize these values to sum to 1 (or 100% if multiplied by 100)
    return {emotion: round(value / total * 100, 2) if total > 0 else 0 for emotion, value in mapped_emotions.items()}

def visualize_emotions(file_path_trump, file_path_obama): # Load emotion data from files
    try:
        with open(file_path_trump, 'r') as file:
            trump_data = json.load(file)
        with open(file_path_obama, 'r') as file:
            obama_data = json.load(file)
    except FileNotFoundError as e:
        print(f"Error opening file: {e}")
        return

    # Extract the emotions list from the data
    emotions_trump = trump_data['emotions']
    emotions_obama = obama_data['emotions']
    
    # Calculate the average emotion scores
    average_emotions_trump = {emotion: sum(d[emotion] for d in emotions_trump) / len(emotions_trump) for emotion in emotions_trump[0]}
    average_emotions_obama = {emotion: sum(d[emotion] for d in emotions_obama) / len(emotions_obama) for emotion in emotions_obama[0]}

    # Data preparation
    emotions = list(average_emotions_trump.keys())
    scores_trump = [average_emotions_trump[emotion] for emotion in emotions]
    scores_obama = [average_emotions_obama[emotion] for emotion in emotions]

    # Plotting
    x = list(range(len(emotions)))  # Now x is a list of integers
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    # Use list comprehension to adjust x positions
    rects1 = ax.bar([xi - width/2 for xi in x], scores_trump, width, label='Trump', color='tomato')  
    rects2 = ax.bar([xi + width/2 for xi in x], scores_obama, width, label='Obama', color='dodgerblue')  

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Emotion Scores (%)')
    ax.set_title('Emotion Scores by Speaker')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions)
    ax.legend()

    def autolabel(rects): # Attach a text label above each bar in *rects*, displaying its height.
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()

#
def process_text(data): # Process and categorize text by POS with frequency count, excluding specific words."""
    exclude_words = {'barack', 'obama', 'president', 'donald', 'trump'}
    pos_categories = defaultdict(Counter)
    for entry in data:
        doc = nlp(entry['text'])
        for token in doc:
            if token.text.lower() in exclude_words or token.pos_ in ['PUNCT', 'SPACE', 'SYM']:
                continue
            pos_categories[token.pos_].update([token.text.lower()])
    return {pos: dict(words) for pos, words in pos_categories.items()}

def save_data(data, file_path): # Save processed data to a JSON file, adapted to handle dictionary data."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def load_data(file_path): # Load data from a JSON file """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
#
def get_top_adjectives(data, top_n=10):
    # Check if 'ADJ' key exists in the data, then extract adjectives
    adjectives = data.get('ADJ', {})
    return sorted(adjectives.items(), key=lambda x: x[1], reverse=True)[:top_n]

def visualize_top_adjectives_side_by_side(file_path_trump, file_path_obama):
    trump_data = load_data(file_path_trump)
    obama_data = load_data(file_path_obama)

    top_adjectives_trump = get_top_adjectives(trump_data)
    top_adjectives_obama = get_top_adjectives(obama_data)

    # Splitting words and counts for plotting
    adjectives_trump, counts_trump = zip(*top_adjectives_trump) if top_adjectives_trump else ([], [])
    adjectives_obama, counts_obama = zip(*top_adjectives_obama) if top_adjectives_obama else ([], [])

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # 1 row, 2 columns of graphs

    # Trump's graph on the left
    axs[0].barh(range(len(adjectives_trump)), counts_trump, color='tomato', align='center')
    axs[0].set(yticks=range(len(adjectives_trump)), yticklabels=adjectives_trump)
    axs[0].set_title('Top Adjectives Used by Trump')
    axs[0].invert_xaxis()  # Invert x-axis to have the bars grow towards the center

    # Obama's graph on the right
    axs[1].barh(range(len(adjectives_obama)), counts_obama, color='dodgerblue', align='center')
    axs[1].set(yticks=range(len(adjectives_obama)), yticklabels=adjectives_obama)
    axs[1].set_title('Top Adjectives Used by Obama')

    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
#
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
#
# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to normalize text
def normalize_text(text):
    text = text.replace(u'\u2019', "'").replace(u'\u2018', "'")  # Convert curly apostrophes to straight ones
    text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text)  # Remove times like 12:30
    doc = nlp(text)  # Apply spaCy NER to capitalize proper nouns and abbreviations
    tokens = []
    for token in doc:
        if token.ent_type_:
            tokens.append(token.text_with_ws)  # Preserve entities as they are detected
        else:
            if token.text.isupper() and len(token.text) > 1:  # Handle non-entities
                tokens.append(token.text_with_ws)  # Preserve uppercase abbreviations like USA, U.S., etc.
            else:
                # Apply lowercase to non-entity, non-abbreviation words
                tokens.append(token.text_with_ws.lower())  # Apply lowercase to non-entity, non-abbreviation words

    normalized_text = ''.join(tokens)
    normalized_text = re.sub(r'[^\w\s,.\'-]', '', normalized_text)  # Remove unwanted characters, but keep periods, commas, and apostrophes
    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()  # Replace multiple spaces with a single space and strip leading/trailing spaces

    return normalized_text

# Function to process JSON files
def process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    normalized_data = []
    for entry in data:
        normalized_entry = {'text': normalize_text(entry['text'])}
        normalized_data.append(normalized_entry)
    return normalized_data

# List of files to be processed
json_files = ['trump_speeches.json', 'obama_speeches.json']

# Process and normalize each JSON file
normalized_obama_speeches = process_json('obama_speeches.json')
normalized_trump_speeches = process_json('trump_speeches.json')

print("Preprocessing complete.")

# Function to display topics
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topic_words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        topics.append(topic_words)
        print(f"Topic {topic_idx + 1}:")
        print(topic_words)
    return topics

# Function to split text into chunks
def split_into_chunks(text, chunk_size=100):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Extract text content from each speech and split into chunks
obama_texts = [speech['text'] for speech in normalized_obama_speeches]
trump_texts = [speech['text'] for speech in normalized_trump_speeches]

obama_chunks = []
for text in obama_texts:
    obama_chunks.extend(split_into_chunks(text))

trump_chunks = []
for text in trump_texts:
    trump_chunks.extend(split_into_chunks(text))

# Combine the speech chunks into a single dataset
combined_chunks = obama_chunks + trump_chunks
labels = ['Obama'] * len(obama_chunks) + ['Trump'] * len(trump_chunks)

# Vectorize the combined text data
vectorizer = CountVectorizer(stop_words='english', max_features=10000, max_df=0.95, min_df=2)
X = vectorizer.fit_transform(combined_chunks)

# Apply LDA
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(X)

print("Combined Topics:")
combined_topics = display_topics(lda, vectorizer.get_feature_names_out(), 10)

# Transform data to get topic distributions
topic_distributions = lda.transform(X)

# Assign the dominant topic to each chunk
dominant_topics = topic_distributions.argmax(axis=1)

# Create a DataFrame for visualization
df = pd.DataFrame({
    'Chunk': combined_chunks,
    'Label': labels,
    'Dominant Topic': dominant_topics
})

# Visualize the topic distribution
topic_counts = df.groupby(['Label', 'Dominant Topic']).size().unstack().fillna(0)

topic_counts.plot(kind='bar', stacked=True, figsize=(10, 7))
plt.title('Topic Distribution in Obama and Trump Speeches')
plt.xlabel('Speaker')
plt.ylabel('Number of Chunks')
plt.show()
#
def main():
    json_files = ['trump_speeches.json', 'obama_speeches.json']
    for file_path in json_files:
        normalized_data = process_json(file_path)
        output_file_path = 'normalized_' + file_path
        save_normalized_data(normalized_data, output_file_path)
        print(f"Normalized data saved to {output_file_path}")
        
    files_and_outputs = {
        'normalized_trump_speeches.json': 'trump_data.json',
        'normalized_obama_speeches.json': 'obama_data.json',
    }
    for input_path, output in files_and_outputs.items():
        data = load_data(input_path)
        pos_tags = process_text(data)
        pos_data_file = output.replace('_data.json', '_pos_data.json')
        save_data(pos_tags, pos_data_file)

        # Extract texts from the data for get_catchphrases function
        texts = [entry['text'] for entry in data]  # assuming each entry in data is a dictionary with a 'text' key
        result = {
            'catchphrases': get_catchphrases(texts, min_n=3, max_n=None, top_n=10),
            'longest_sentences': get_longest_sentences(data),
            'emotions': [analyze_emotions(entry['text']) for entry in data]
        }
        save_data(result, output)
        print(f"All data saved for {input_path} in {output}")

    trump_data = load_data('trump_data.json')
    obama_data = load_data('obama_data.json')
    
    # Assuming the structure of the JSON file has a dictionary with a key 'catchphrases' that contains a list of phrases
    trump_catchphrases = Counter(trump_data['catchphrases'])
    obama_catchphrases = Counter(obama_data['catchphrases'])
    
    # Visualize 
    visualize_catchphrases_side_by_side(trump_catchphrases, obama_catchphrases, 'catchphrases_comparison.png')
    visualize_emotions('trump_data.json', 'obama_data.json')
    visualize_top_adjectives_side_by_side('trump_pos_data.json', 'obama_pos_data.json')

    # Topic Modeling part
    # Load normalized texts
    normalized_obama_speeches = load_data('normalized_obama_speeches.json')
    normalized_trump_speeches = load_data('normalized_trump_speeches.json')

    # Extract text content from each speech and split into chunks
    obama_texts = [speech['text'] for speech in normalized_obama_speeches]
    trump_texts = [speech['text'] for speech in normalized_trump_speeches]

    obama_chunks = []
    for text in obama_texts:
        obama_chunks.extend(split_into_chunks(text))

    trump_chunks = []
    for text in trump_texts:
        trump_chunks.extend(split_into_chunks(text))

    # Combine the speech chunks into a single dataset
    combined_chunks = obama_chunks + trump_chunks
    labels = ['Obama'] * len(obama_chunks) + ['Trump'] * len(trump_chunks)

    # Vectorize the combined text data
    vectorizer = CountVectorizer(stop_words='english', max_features=10000, max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(combined_chunks)

    # Apply LDA
    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    lda.fit(X)

    print("Combined Topics:")
    combined_topics = display_topics(lda, vectorizer.get_feature_names_out(), 10)

    # Transform data to get topic distributions
    topic_distributions = lda.transform(X)

    # Assign the dominant topic to each chunk
    dominant_topics = topic_distributions.argmax(axis=1)

    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Chunk': combined_chunks,
        'Label': labels,
        'Dominant Topic': dominant_topics
    })

    # Visualize the topic distribution
    topic_counts = df.groupby(['Label', 'Dominant Topic']).size().unstack().fillna(0)

    topic_counts.plot(kind='bar', stacked=True, figsize=(10, 7))
    plt.title('Topic Distribution in Obama and Trump Speeches')
    plt.xlabel('Speaker')
    plt.ylabel('Number of Chunks')
    plt.show()

if __name__ == "__main__":
    main()
