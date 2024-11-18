from transformers import BertTokenizer, BertModel
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np

# Step 1: Load Text File
file_path = '/Users/tejas/Desktop/LLM_Analysis/1901.07176.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and truncate or split the input text
tokenized_text = tokenizer(content, return_tensors='pt', truncation=True, max_length=512)

# You may need to loop through chunks of the text if it's too long
# Example: splitting into chunks of 512 tokens
chunk_size = 512
chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

embeddings_list = []

for chunk in chunks:
    tokens = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512)
    embeddings = model(**tokens).last_hidden_state.mean(dim=1).detach().numpy()
    embeddings_list.extend(embeddings)

# Step 3: User Question Input
user_question = input("Enter your question: ")

# Use transformers pipeline for feature extraction
from transformers import pipeline
feature_extraction_pipeline = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
user_vector = feature_extraction_pipeline(user_question)[0]
user_vector_mean = np.mean(user_vector, axis=0)

# Step 4: KNN for Finding Closest Vector
knn_model = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embeddings_list)
_, closest_index = knn_model.kneighbors(user_vector_mean.reshape(1, -1))

# Step 5: User Feedback and Discretion
closest_answer = content.split('\n')[closest_index[0][0]]
print(f"Closest answer: {closest_answer}")

user_feedback = input("Is the answer correct? (yes/no): ")

if user_feedback.lower() == 'no':
    user_correction = input("Provide the correct answer: ")
    # Handle user correction, for example, update the dataset with the correct answer
    content_lines = content.split('\n')
    content_lines[closest_index[0][0]] = user_correction
    updated_content = '\n'.join(content_lines)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)
    print("Dataset updated with user correction.")
else:
    print("Thanks for the feedback!")
