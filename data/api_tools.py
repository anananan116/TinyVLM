import json
import pprint
import copy
import time
import argparse
from openai import OpenAI
from creditials import OpenAI_api_key
from concurrent.futures import ThreadPoolExecutor, as_completed

SYSTEM_PROMPT = """
The task is to refine and enhance answers in a visual instruction dataset for consistency, accuracy, completeness, and adherence to instructions, while also ensuring that reasoning steps are included when appropriate.

Please use the provided format to structure the output properly, merging information for a complete and enriched response to each question.

# Requirements:

1. **Consistency in Answer Format**: When the question is simple and direct (e.g. "What is the man holding?"), adjust the answer to follow a complete, descriptive format that aligns with the question—such as "The man is holding a [object]."
   
2. **Intermediate Reasoning for Complex Questions**: For questions that require analysis or deduction, make sure the answer includes the reasoning process before jumping to the conclusion. This should convey how the answer was derived.

3. **Provide Detailed Answers**: Ensure all answers are detailed enough to fully cover the scope of the question, adding complementary information if necessary to ensure completeness.

4. **Fix Inaccurate Answers**: If an existing answer is inaccurate, rewrite it to accurately reflect the content of the image.

5. **Follow the Style from Instructions**: Make sure that answers adhere closely to the format or style requested in the instructions whenever such a style is provided.

6. **Unify Question and Instruction**: In cases where the distinction between "instruction" and "question" is mixed up or missing entirely, clearly separate and provide both fields as needed.

# Steps:

1. **Analyze the Task**: Review the instruction, question, image, and given answer. Identify if any revisions need to be made based on consistency, accuracy, adherence to instruction, or detail.
2. **Revise Each Answer**: Apply the following changes if relevant:
   - Reformat the answer according to the question type (e.g., simple direct questions should have clean phrasing, more detailed questions should include reasoning).
   - Include any missing intermediary reasoning steps.
   - Add detail where the answer appears incomplete.
   - Correct any inaccuracies in the original answer.
3. **Separate and Rewrite Instruction and Question**: Ensure the instruction and question are not mixed and represent separate and valid content.
4. **Generate Output in JSON Format**: Provide the revised version, ensuring all fields (`instruction`, `question`, `answer`, `full_answer`, and `answer_only`) are accurately filled.

# Output Format

Output the completed task as structured JSON in the following format:

```json
{
    "instruction": "<str>",
    "question": "<str>",
    "answer": "<str>",
    "full_answer": "<str>",
    "answer_only": "<str>"
}
```
- **instruction**: The instruction that guides how the question should be answered.
- **question**: The specific question about the image.
- **full_answer**: A complete and enriched answer. For simple questions, the full answer should fully state the fact; for complex questions, include reasoning followed by the answer.
- **answer_only**: The final answer in a concise form—can be a word, phrase, or specific answer.

# Examples

### Example 1: Complex Question
Input: 
- Instruction: Analyze the scene and provide contextual reasoning.
- Question: "Why is the woman smiling while looking at the cake?"
- Original Answer: "Because she's happy."

Output:
```json
{
    "instruction": "Analyze the scene and provide contextual reasoning.",
    "question": "Why is the woman smiling while looking at the cake?",
    "full_answer": "The woman appears to be smiling while looking at the cake because it might be her birthday or a special celebration. The cake is placed prominently with lit candles, which suggests a festive event. Therefore, she is smiling because she is happy to celebrate this special occasion.",
    "answer_only": "celebration"
}
```

### Example 2: Simple Question
Input:
- Instruction: Identify the object in the person's hand.
- Question: "What is the child holding?"
- Original Answer: "A balloon."

Output:
```json
{
    "instruction": "Identify the object in the person's hand.",
    "question": "What is the child holding?",
    "full_answer": "The child is holding a blue balloon with a shiny surface.",
    "answer_only": "blue balloon"
}
```

### Example 3: Where Question and Instruction are Mixed
Input: 
- Mixed Instruction and Question: "Analyze carefully. What is the woman doing?"
- Original Answer: "She is dancing."

Output:
```json
{
    "instruction": "Analyze the activity the woman is performing.",
    "question": "What is the woman doing?",
    "full_answer": "The woman is dancing gracefully, performing what appears to be a traditional routine involving fluid arm movements.",
    "answer_only": "dancing"
}
```

# Notes

- Always provide a detailed reasoning process when applicable, and ensure that answers are both complete and concise.
- For mixed questions and instructions, rely on logical separation and rewrite to provide equivalent instruction and question separately.
- Be cautious of images with ambiguous content; only provide details supported by clear visual evidence."""

TEMPLATE = {"custom_id": "", 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {"model": "gpt-4o-mini", 
                     "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
                     "max_tokens": 512}}

def send_request(queries, cache_num = 0, index = 0):
    with open(f"./data/cache/queries_temp_{cache_num}_index_{index}.jsonl", "w") as f:
        for entry in queries:
            json.dump(entry, f)
            f.write('\n')
    client = OpenAI(api_key = OpenAI_api_key)
    batch_input_file = client.files.create(
    file=open(f"./data/cache/queries_temp_{cache_num}_index_{index}.jsonl", "rb"),
    purpose="batch"
    )
    
    batch_input_file_id = batch_input_file.id

    batch_object = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "TEST"
        }
    )
    id = batch_object.id
    return id


def report_status(start_time, last_report_time = 0, index = 0):
    if time.time() - last_report_time > 30:
        print(f"Batch index {index}: Waiting for OpenAI API... {time.time() - start_time} seconds elapsed.")
        last_report_time = time.time()
    return last_report_time
    

def wait_for_completion_text(batch_id, index):
    last_report_time = 0
    start_time = time.time()
    client = OpenAI(api_key = OpenAI_api_key)
    batch_object = client.batches.retrieve(batch_id)
    while batch_object.status != "completed":
        batch_object = client.batches.retrieve(batch_id)
        report_status(start_time, last_report_time, index)
        if batch_object.status == "failed":
            raise ValueError("Batch failed")
        time.sleep(60)
    output_file_id = batch_object.output_file_id
    output_file = client.files.content(output_file_id)
    output = output_file.text.split('\n')[:-1]
    output_content = {json.loads(x)["custom_id"]: json.loads(x)["response"]["body"]["choices"][0]["message"]["content"] for x in output}
    return output_content

def wait_for_completion_structured(batch_id, index):
    last_report_time = 0
    start_time = time.time()
    client = OpenAI(api_key = OpenAI_api_key)
    batch_object = client.batches.retrieve(batch_id)
    while batch_object.status != "completed":
        report_status(start_time, last_report_time, index)
        batch_object = client.batches.retrieve(batch_id)
        if batch_object.status == "failed":
            raise ValueError("Batch failed")
        time.sleep(60)
    output_file_id = batch_object.output_file_id
    output_file = client.files.content(output_file_id)
    output = output_file.text.split('\n')[:-1]
    output_content = {}
    for x in output:
        try:
            loaded = json.loads(x)
            output_content[loaded["custom_id"]] = json.loads(loaded["response"]["body"]["choices"][0]["message"]["content"])
        except:
            pass
    return output_content

def send_batch(queries, output_type, index):
    if output_type == "text":
        return wait_for_completion_text(send_request(queries, index = index), index)
    elif output_type == "structured":
        return wait_for_completion_structured(send_request(queries, index = index), index)
    else:
        raise ValueError("output_type not supported")

def send_all_queries(queries, output_type="text", batch_size=50000, max_workers=5):
    all_output = {}
    futures = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, index in zip(range(0, len(queries), batch_size), range(len(queries)//batch_size + 1)):
            batch = queries[i:i + batch_size]
            print(f"Sending queries {i} to {i + batch_size}")
            futures.append(executor.submit(send_batch, batch, output_type, index))

        for future in as_completed(futures):
            try:
                output = future.result()
                for k, v in output.items():
                    all_output[k] = v
            except Exception as e:
                print(f"Error processing batch: {e}")

    return all_output

def main():
    prompt = "Here's a question and the answer to that question. Please extend the answer so that it is a complete response suitable for a chatbot. Please just respond with the augmented answer ONLY!"
    question = "What's the object on the left of the image?"
    answer = "A red apple."
    queries = copy.deepcopy(TEMPLATE)
    queries["custom_id"] = "0"
    queries["body"]["messages"].append({"role": "user", "content": f"{prompt} Question: {question}. Answer: {answer}"})
    queries = [queries]
    print(send_all_queries(queries, "text"))

if __name__ == "__main__":
    main()