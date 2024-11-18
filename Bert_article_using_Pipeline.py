from transformers import pipeline

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def question_answering(article, questions):
    # Load the question-answering pipeline
    qa_pipeline = pipeline("question-answering")

    for question in questions:
        # Perform question-answering
        result = qa_pipeline(question=question, context=article)

        # Extract the answer
        answer = result['answer']
        print(f"Question: {question}\nAnswer: {answer}\n")

# Example usage
file_path = "MoralEconomics.txt"
article_text = read_text_file(file_path)
questions_to_ask = ["What is the main idea?", "Who is this conversation between?"]

question_answering(article_text, questions_to_ask)
