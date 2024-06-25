import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def compute_similarity(text1, text2):
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_scores.item()

# Load speeches from JSON files
obama_speeches = read_json('/Users/ototchigladze/Desktop/Obama Trump/normalized_obama_speeches.json')
trump_speeches = read_json('/Users/ototchigladze/Desktop/Obama Trump/normalized_trump_speeches.json')

# Extract full speech texts
obama_texts = [speech['text'] for speech in obama_speeches if 'text' in speech]
trump_texts = [speech['text'] for speech in trump_speeches if 'text' in speech]

# Compute similarities between each pair of speeches
similarity_results = []

for obama_text in obama_texts:
    for trump_text in trump_texts:
        similarity_score = compute_similarity(obama_text, trump_text)
        similarity_results.append({
            'obama_speech': obama_text[:100],  # Truncated for readability
            'trump_speech': trump_text[:100],  # Truncated for readability
            'similarity_score': similarity_score
        })

# Sort results by similarity score
similarity_results = sorted(similarity_results, key=lambda x: x['similarity_score'], reverse=True)

# Display top 5 most similar speech pairs
for result in similarity_results[:5]:
    print(f"Obama Speech (truncated): {result['obama_speech']}...")
    print(f"Trump Speech (truncated): {result['trump_speech']}...")
    print(f"Similarity Score: {result['similarity_score']:.4f}")
    print("-" * 80)

