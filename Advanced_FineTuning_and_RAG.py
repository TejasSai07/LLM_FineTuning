import torch
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    AdamW, get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import numpy as np
import json
from typing import List, Dict, Tuple
import nltk
import pickle

# Download necessary NLTK data
punkt_path = '/Users/tejas/nltk_data/tokenizers/punkt/punkt/english.pickle'
with open(punkt_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Use the loaded tokenizer for sentence tokenization


class DocumentLoader:
    @staticmethod
    def load_and_split(filepath: str, chunk_size: int = 3) -> List[str]:
        """
        Load document and split into chunks of sentences
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Use the manually loaded tokenizer
        s = tokenizer.tokenize(text)
        
        # Create chunks of s
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks


class QADataset(Dataset):
    def __init__(self, questions, contexts, answers, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.questions[idx],
            self.contexts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert answer text to token positions
        answer_start_char = self.answers[idx]['answer_start']
        answer_text = self.answers[idx]['text']
        
        # Find the token positions that correspond to the character positions
        context_encoding = self.tokenizer(self.contexts[idx], return_offsets_mapping=True)
        offset_mapping = context_encoding.offset_mapping
        
        start_position = end_position = 0
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= answer_start_char < end:
                start_position = idx
            if start < answer_start_char + len(answer_text) <= end:
                end_position = idx
                break
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': torch.tensor(start_position),
            'end_positions': torch.tensor(end_position)
        }

class RAGEnhancedQA:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.model_name).to(device)
        
        # Initialize embedding model for retrieval
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.index = None
        self.passages = []

    def load_documents_from_file(self, filepath: str):
        """Load and index documents from a text file"""
        doc_loader = DocumentLoader()
        chunks = doc_loader.load_and_split(filepath)
        self.index_documents(chunks)
        print(f"Loaded and indexed {len(chunks)} passages from {filepath}")

    def index_documents(self, documents):
        """Create searchable index from documents"""
        self.passages = documents
        print("Encoding documents...")
        embeddings = self.embedding_model.encode(documents, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()

        # Create FAISS index
        self.index = IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print("Documents indexed successfully")

    def retrieve_relevant_passages(self, query, top_k=5):
        """
        Retrieve top_k relevant passages for the given query using FAISS index.
        """
        # Encode the query using the same embedding model
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy()

        # Reshape to 2D if it's 1D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Perform similarity search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve the top_k relevant passages
        relevant_passages = [self.passages[idx] for idx in indices[0] if idx != -1]

        return relevant_passages


    def answer_question(self, question, max_length=384):
        """Answer question using RAG approach"""
        if self.index is None:
            raise ValueError("No documents have been indexed. Please index documents first.")

        # Retrieve relevant passages
        relevant_passages = self.retrieve_relevant_passages(question)
        combined_context = " ".join(relevant_passages)

        # Generate answer using QA model
        inputs = self.tokenizer(
            question,
            combined_context,
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]
            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            answer_tokens = tokens[start_idx:end_idx + 1]
            answer = self.tokenizer.convert_tokens_to_string(answer_tokens).strip()

            if not answer or answer.isspace():
                answer = "Could not find a relevant answer in the context."

        return answer, relevant_passages
    def prepare_training_data(self, squad_file: str) -> QADataset:
        """Prepare training data from SQuAD format JSON"""
        with open(squad_file, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)
        
        questions = []
        contexts = []
        answers = []
        
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    if qa['answers']:  # Only use examples with answers
                        questions.append(qa['question'])
                        contexts.append(context)
                        answers.append({
                            'text': qa['answers'][0]['text'],
                            'answer_start': qa['answers'][0]['answer_start']
                        })
        
        return QADataset(questions, contexts, answers, self.tokenizer)
    def fine_tune(self, train_dataset, epochs=3, batch_size=8, learning_rate=5e-5):
        """Fine-tune the QA model on custom dataset"""
        print("Starting fine-tuning process...")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.qa_model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,  # 10% of total steps for warmup
            num_training_steps=total_steps
        )
        
        # Training loop
        self.qa_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                # Forward pass
                outputs = self.qa_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.qa_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Print progress
                if (batch_idx + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")


def main():
    # Initialize enhanced QA system
    qa_system = RAGEnhancedQA()
    
    # Load and index documents from Moral_economics.txt
    qa_system.load_documents_from_file('MoralEconomics.txt')
    
    # Example of fine-tuning (if you have SQuAD format data)
    squad_file = "squad_manchester_united.json"
    train_dataset = qa_system.prepare_training_data(squad_file)
    qa_system.fine_tune(train_dataset, epochs=3)
    
    # Test the system
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        answer, relevant_passages = qa_system.answer_question(question)
        print(f"\nAnswer: {answer}")
        print("\nRelevant passages:")
        for i, passage in enumerate(relevant_passages, 1):
            print(f"{i}. {passage}")

if __name__ == "__main__":
    main()