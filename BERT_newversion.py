import torch

from transformers import BertTokenizer, BertForQuestionAnswering

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

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
            # print(current_chunk)
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def answer_question(question, context, max_context_length=384):
    if not isinstance(context, str):
        raise ValueError("The 'context' parameter must be a string.")
    # print(context)
    # Concatenate all chunks into a single context
    # context = " ".join(context)

    # Tokenize the question and full context
    input_ids = tokenizer.encode(question, context, max_length=max_context_length, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Prepare input with segment IDs for question and context
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)

    # Use the BERT model to predict start and end positions of the answer
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)



    if answer_start <= answer_end:
        # Extract the answer from tokens
        answer = tokens[answer_start:answer_end+1]
        answer = tokenizer.convert_tokens_to_string(answer)
        return answer
    else:
        return "I couldn't find the answer to your question."

# Example usage
file_path = "/Users/tejas/Desktop/LLM_Analysis/MoralEconomics.txt"
text = load_text(file_path)
print(type(text))  # Ensure that text is a string

questions = "The Annual Review of Economics is a relatively new member of ?"
max_sequence_length = 512  # Adjust based on the maximum sequence length supported by your BERT model


example_question = 'Where does the club compete?'
filename = "/Users/tejas/Desktop/LLM_Analysis/MoralEconomics.txt"
with open(filename, 'r') as file:
        text = file.read()
example_context = text
# example_context = "Manchester United Football Club, commonly referred to as Man United (often stylised as Man Utd), or simply United, is a professional football club based in Old Trafford, Greater Manchester, England. The club competes in the Premier League, the top division in the English football league system. Nicknamed the Red Devils, they were founded as Newton Heath LYR Football Club in 1878, but changed their name to Manchester United in 1902. After a spell playing in Clayton, Manchester, the club moved to their current stadium, Old Trafford, in 1910Domestically, Manchester United have won a record 20 top-flight league titles, 12 FA Cups, six League Cups and a record 21 FA Community Shields. In international football, they have won the European Cup/UEFA Champions League three times, and the UEFA Europa League, the UEFA Cup Winners' Cup, the UEFA Super Cup, the Intercontinental Cup and the FIFA Club World Cup once each.[5][6] In 1968, under the management of Matt Busby, 10 years after eight of the club's players were killed in the Munich air disaster, they became the first English club to win the European Cup. Sir Alex Ferguson is the club's longest-serving and most successful manager, winning 38 trophies, including 13 league titles, five FA Cups, and two Champions League titles between 1986 and 2013.[7][8] In the 1998–99 season, under Ferguson, the club became the first in the history of English football to achieve the continental treble of the Premier League, FA Cup and UEFA Champions League.[9] In winning the UEFA Europa League under José Mourinho in 2016–17, they became one of five clubs to have won the original three main UEFA club competitions (the Champions League, Europa League and Cup Winners' Cup).Manchester United is one of the most widely supported football clubs in the world[10][11] and has rivalries with Liverpool, Manchester City, Arsenal and Leeds United. Manchester United was the highest-earning football club in the world for 2016–17, with an annual revenue of €676.3 million,[12] and the world's third-most-valuable football club in 2019, valued at £3.15 billion ($3.81 billion).[13] After being floated on the London Stock Exchange in 1991, the club was taken private in 2005 after a purchase by American businessman Malcolm Glazer valued at almost £800 million, of which over £500 million of borrowed money became the club's debt.[14] From 2012, some shares of the club were listed on the New York Stock Exchange, although the Glazer family retains overall ownership and control of the club.ed equalled their own record for the biggest win in Premier League history with a 9–0 win over Southampton on 2 February 2021,[87] but ended the season with defeat on penalties in the UEFA Europa League final against Villarreal, going four straight seasons without a trophy.[88] On 20 November 2021, Solskjær left his role as manager.[89] Former midfielder Michael Carrick took charge for the next three games, before the appointment of Ralf Rangnick as interim manager until the end of the season.[90]On 21 April 2022, Erik ten Hag was appointed as the manager from the end of the 2021–22 season, signing a contract until June 2025 with the option of extending for a further year.[91] Ten Hag won Manchester United the 2022–23 EFL Cup against Newcastle United, winning 2–0.[92] On 5 March 2023, the club suffered their joint-heaviest defeat, losing 7–0 to rivals Liverpool at Anfield.[93]"

result = answer_question(example_question, example_context)
print(result)
