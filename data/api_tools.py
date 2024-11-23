import json
import pprint
import copy
import time
import argparse
from openai import OpenAI
from creditials import OpenAI_api_key
from concurrent.futures import ThreadPoolExecutor, as_completed

SYSTEM_PROMPT = "You are a helpful assistant for data aguementation."

TEMPLATE = {"custom_id": "", 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {"model": "gpt-4o-mini", 
                     "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
                     "max_tokens": 128}}

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