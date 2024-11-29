import pandas as pd
import random
CAPTION_PROMPTS = [
    "Describe this image.",
    "Write a caption for this image.",
    "What does this image show?",
    "Generate a caption describing this image.",
    "Provide a description of this image.",
    "What's depicted in this image?",
    "Write a description for this image.",
    "Generate text describing this image.",
    "Create a caption for this image.",
    "What can you see in this image?",
    "Describe what you observe in this image.",
    "Write text explaining this image.",
    "Generate a description of this image.",
    "What is shown in this image?",
    "Provide a caption for this image.",
    "Explain what this image depicts.",
    "Write what you see in this image.",
    "Describe the contents of this image.",
    "Generate text about this image.",
    "What appears in this image?"
]
def get_random_prompt():
    return random.choice(CAPTION_PROMPTS)

INSTRUCTION = "Caption the image."

def adapt_data(input_path, output_path):
    df = pd.read_csv(input_path)
    new_df = pd.DataFrame(columns=["instruction", "inputs", "outputs", "image_path"])
    new_df["outputs"] = df["capsfusion"]
    new_df["inputs"] = [get_random_prompt() for _ in range(len(df))]
    new_df["instruction"] = [INSTRUCTION] * len(df)
    new_df["image_path"] = "images/" + df["identifier"] + ".jpg"
    new_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    adapt_data("data/image_metadata_complete.csv", "data/pretrain_data.csv")