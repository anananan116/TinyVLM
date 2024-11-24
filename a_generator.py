import api_tools as tool
import pandas as pd
import copy
import os
import pandas as pd

# Define the questions and one-word answers for testing 
# data = {
#     "question": [
#         "What is the capital of France?",
#         "What is the largest planet in our solar system?",
#         "What is the square root of 16?",
#         "Who wrote 'Romeo and Juliet'?",
#         "What is the chemical symbol for water?"
#     ],
#     "multiple_choice_answer": [
#         "Paris",
#         "Jupiter",
#         "4",
#         "Shakespeare",
#         "H2O"
#     ]
# }

# df = pd.DataFrame(data)

def turn_df_to_df(df):
    #List of queries
    qs = []
    #default system prompt
    prompt = "Here's a question and the answer to that question. Please extend the answer so that it is a complete response suitable for a chatbot. Please just respond with the augmented answer ONLY!"
    #list of questions
    questions = df['question'].tolist()
    #list of single-word answer
    answers = df['multiple_choice_answer'].tolist()

    for i in range(len(questions)):
        query = copy.deepcopy(tool.TEMPLATE)
        query["custom_id"] = str(i)
        question = questions[i]
        answer = answers[i]
        query["body"]["messages"].append({"role": "user", "content": f"{prompt} Question: {question}. Answer: {answer}"})
        qs.append(query)
    output = tool.send_all_queries(qs, "text")
    df_out = pd.DataFrame(list(output.items()), columns=["custom_id", "answer"])
    df['complete_answer'] = df_out["answer"]
    return df
    

# directory_path = './data/cache/'
# Create the directory, including any missing parent directories
# os.makedirs(directory_path, exist_ok=True)
# turn_df_to_df(df)
# print(df['complete_answer'][0])
