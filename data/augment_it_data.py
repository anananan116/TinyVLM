import pandas as pd
from api_tools import TEMPLATE, send_all_queries
import copy
import base64
from PIL import Image
from io import BytesIO
from openai.lib._parsing import type_to_response_format_param
from pydantic import BaseModel

class VLMData(BaseModel):
    instruction: str
    question: str
    full_answer: str
    answer_only: str

result_format = type_to_response_format_param(VLMData)

def process_image_to_base64(image_path: str) -> str:
    """
    Opens a JPG image, resizes it if the longer side is greater than 512,
    and encodes it to a base64 string.
    
    Parameters:
        image_path (str): Path to the JPG image file.

    Returns:
        str: Base64-encoded string of the processed image.
    """
    # Open the image
    with Image.open(image_path) as img:
        # Get the original dimensions
        original_width, original_height = img.size
        # Determine the scale factor to resize the longer side to 512
        if max(original_width, original_height) > 512:
            scale_factor = 512 / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            # Resize the image
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
        
        # Convert the image to bytes and encode to base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        base64_string = base64.b64encode(img_bytes).decode("utf-8")
    
    return base64_string


df = pd.read_csv('data/it_train.csv')
df = df[:1000]

all_queries = []

for i, sample in enumerate(df):
    query = copy.deepcopy(TEMPLATE)
    query["custom_id"] = str(i)
    base64_image = process_image_to_base64(sample['image_path'])
    query["body"]["messages"].append(
        {"role": "user", 
         "content": [
             {"type": "text",
              "text": f"Instruction: {sample['instruction']}.\nInputs: {sample['inputs']}.\nAnswer: {sample['outputs']}\n"
              },
             {"type": "image",
                "url": f"data:image/jpeg;base64,{base64_image}"
             }
             ]
         })
    query["body"]["response_format"] = result_format
    all_queries.append(query)

output = send_all_queries(all_queries, "structured")