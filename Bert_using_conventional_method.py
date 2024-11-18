import torch
from transformers import BertTokenizer, BertForQuestionAnswering

def load_text(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def split_context(context, max_context_length):
    words = context.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + len(current_chunk) <= max_context_length:
            current_length += len(word)
            current_chunk.append(word)
        else:
            print(current_chunk)
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def answer_question(question, context, max_context_length=384):
    if not isinstance(context, str):
        raise ValueError("The 'context' parameter must be a string.")
    context_chunks = split_context(context, max_context_length)
    all_answers = []

    for chunk in context_chunks:
        input_ids = tokenizer.encode(question, chunk, max_length=max_context_length, truncation=True)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        sep_index = input_ids.index(tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        assert len(segment_ids) == len(input_ids)
        output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits)
        if answer_start <= answer_end:
            answer = tokens[answer_start:answer_end+1]
            answer = tokenizer.convert_tokens_to_string(answer)
            all_answers.append((answer, output.start_logits[0, answer_start].item()))

    if not all_answers:
        return "I couldn't find the answer to your question."
    else:
        all_answers.sort(key=lambda x: x[1], reverse=True)
        return all_answers[0][0]

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Example usage
file_path = "MoralEconomics.txt"
text = load_text(file_path)
print(type(text))  # Ensure that text is a string

questions = "The Annual Review of Economics is a relatively new member of ?"
max_sequence_length = 512  # Adjust based on the maximum sequence length supported by your BERT model

# Split text into chunks and process each chunk separately
chunks = split_context(text, max_sequence_length)
for chunk in chunks:
    result = answer_question(questions, chunk)
    print(result)
