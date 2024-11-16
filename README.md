# TinyVLM

To try out our model:

Install dependencies:

```bash
pip install transformers
```

NOTE that the model on hugging face is still a place holder!

Perform inference (a minimum of 4G free vram is needed to perform inference):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
import torch 

model = AutoModelForCausalLM.from_pretrained(
    "anananan116/TinyVLM",
    trust_remote_code = True,
    torch_dtype=torch.float16,
    ).to('cuda').eval()
tokenizer = AutoTokenizer.from_pretrained("anananan116/TinyVLM")

# `<IMGPLH>` is the image placeholder which will be replaced by image embeddings. 
# the number of `<IMGPLH>` should be equal to the number of input images

prompt = "Here's an image: <IMGPLH>Create a caption for this image."
image = Image.open(requests.get('https://github.com/anananan116/TinyVLM/blob/main/test.png?raw=true',stream=True).raw)
inputs = model.prepare_input_ids_for_generation([prompt], [image], tokenizer)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs['input_ids'].to("cuda"), 
        attention_mask=inputs['attention_mask'].to("cuda"), 
        encoded_image = inputs["encoded_image"], 
        max_new_tokens=128, 
        do_sample=True
    )

output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

## Model at A Glance

### Tiny Vision-Language Model

In this project, we focus on training a vision language model. A vision language model is a neural network architecture that bridges the gap between visual and textual information, enabling AI systems to understand, describe, and reason about images in natural language. These models combine computer vision capabilities with natural language processing to perform tasks such as image captioning, visual question answering (VQA), and image-based reasoning. By learning joint representations of images and text, these models can establish meaningful connections between visual elements and linguistic descriptions, making them fundamental to applications in AI assistants, content analysis, and accessibility tools.

### Training Setup

Our training approach follows a carefully designed two-stage process:

1. **Initial Image Captioning Phase**
   - Train on a large-scale caption dataset, prioritizing breadth over precision
   - This foundational stage allows the model to:
     - Develop basic visual recognition capabilities
     - Learn broad visual-linguistic associations
     - Build a comprehensive vocabulary for describing visual content
     - Understand common objects, actions, and scenes in images

2. **Fine-tuning Phase**
   - Instruction tuning on:
     - VQA (Visual Question Answering) datasets
     - High-quality, detailed caption datasets
   - This advanced stage enables:
     - Complex reasoning about visual content
     - Handling specific user instructions and queries
     - More nuanced and accurate image descriptions
     - Understanding and responding to diverse user prompts
     - Better alignment with human intent and expectations

The result of the two stages are our First and Final model respectively.

### "Tiny" Model

Our model combines a Llama 3.2 1B language model with a CLIP ViT-L (~400M parameters) vision encoder. This lightweight approach makes the model both practical and useful by reducing computational requirements and deployment costs while maintaining reasonable performance. The smaller footprint enables deployment on edge devices and servers with limited resources, making AI more accessible to a broader range of applications and organizations. This approach demonstrates that effective vision-language models don't necessarily require massive architectures, offering a balance between performance and efficiency.

## Data Exploration and Initial Preprocessing

- Pretraining Data
  - [Image Captioning (Text-Image Pairs)](https://huggingface.co/datasets/BAAI/CapsFusion-120M)
    - This dataset provides over 130 million image links, but we are scaling down. We downloaded the first 5 million rows of the dataset, and of these we will only use the rows where the image link gives a successful response code.
    - All initial images are not uniform in any regard, however during preprocessing, all images will be cropped
    - The dataset is cosist of a synthized caption and an url to the image.
- Instruction Tunning Data
  - [Instruction Tuning #1(VQA) - textVQA](https://textvqa.org/dataset/)
  - [Instruction Tuning #2(VQA) - VQA](https://visualqa.org/download.html)
  - [Instruction Tuning #3(Captioning) - Filtered CC-3M](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)
    - These datasets provide conversation data on diverse tasks. Format varies.

### Pretraining Data

#### Download

We implemented a robust and efficient parallel downloading system for handling large-scale image datasets. The system features automatic retry mechanisms, progress tracking, and chunked processing to handle millions of images. To ensure reliability, it includes comprehensive error handling and generates detailed download statistics and error reports. The system also supports checkpoint-resume functionality, allowing interrupted downloads to continue from where they left off.

Part of the image urls are no longer available anymore as the dataset was published a few years ago. We observe that about 75% of the image can still be downloaded. Here's the download stats for the first 100k images:

![image](assets/download.png)

#### Preprocess

For image preprocessing, we implemented an adaptive approach that intelligently handles images of varying aspect ratios:

- For images with reasonable aspect ratios (close to square), we use center cropping to maintain important visual information. After looking at the distribution of the original aspect ratio of the image, we decided to use 0.6 as the threshold for padding. Below is a histogram showing their distribution.
- For images with extreme aspect ratios, we employ a padding strategy that preserves the entire image content by adding black borders
- All images are resized to a consistent resolution (448x448) while maintaining their quality through high-quality interpolation

![image](assets/aspect_ratio.png)

#### Data Exploration

To take a glance at what kinds of images are in the datasets, we perform image classification and observe the distribution of popular classes. NOTE that this method could be inaccurate since the quality of the labels are not garenteed. However, this is ONLY a data exploration practice, and it has noting to do with our training.

##### Generating labels

To effectively categorize our diverse dataset, we developed a novel approach combining GPT and CLIP to create a rich set of image labels. Rather than relying on traditional pre-defined categories (like ImageNet classes), we first created base categories and used GPT to expand each category with 50-100 semantically related descriptions. This process helped us capture the nuanced variations in our dataset while maintaining categorical coherence. The expanded label set is stored in "labels.py" and serves as the foundation for our zero-shot classification.

##### CLIP for Zero-Shot Classification

We leveraged CLIP (Contrastive Language-Image Pre-training), a powerful vision-language model that can perform zero-shot image classification through natural language descriptions. CLIP works by learning to align image and text representations in a shared embedding space, allowing it to match images with textual descriptions even for categories it hasn't explicitly been trained on.

Our implementation uses the ViT-L/14 CLIP model with the following pipeline:

1. First, we encode our custom text labels into CLIP's embedding space
2. For each image in our dataset:
   - The image is processed through CLIP's vision encoder
   - The similarity between the image embedding and all label embeddings is computed
   - The label with the highest similarity score is assigned to the image
3. The process is optimized with batch processing for efficiency

This approach allows us to categorize images using our custom-generated labels without requiring any additional training, while leveraging CLIP's robust understanding of visual-semantic relationships. The classification results provide both the matched label and a confidence score, allowing us to assess the reliability of the classifications. Below shows the most popular classes of images in our dataset.

![image](assets/image.png)

### Data preparing for instruction tuning

1. Download questions and answers from the dataset
2. Create DataFrame from json data files which include questions and answers
3. Combine questions and answers and create a new DataFrame according to the image_id and answer_id
4. Create answers into a complete sentences
5. Add tags to indicate system prompt, user questions, and answers from the model
6. Combine system prompt, user questions, and answers into one col and label them with corrsponding image_id
7. Output the data as csv file

## Model Architecture

Our model follows a vision-language architecture that combines a modified CLIP visual encoder with a Llama language model, connected through a specialized adaptation layer.

### Visual Encoder

We utilize CLIP's ViT-L/14 vision transformer as our visual backbone but modify its output processing in several key ways:

- Rather than using only the [CLS] token embedding, we extract patch embeddings from the vision transformer
- We implement a spatial pooling strategy that:
  - Removes the [CLS] token
  - Reshapes the patch embeddings into a 2D spatial grid
  - Applies average pooling to reduce the number of patches to a fixed number (64)
  - Maintains the spatial relationships between image regions
- This modified approach provides a more detailed spatial representation of the image compared to the original CLIP's single vector representation

### Large Language Model

Our system builds upon the Llama architecture, a transformer-based language model that uses:

- Multi-head self-attention mechanisms to capture long-range dependencies
- Feed-forward networks for token-level processing
- Layer normalization and residual connections
- A causal attention mask for autoregressive generation

The model processes text using a conventional token embedding layer, but is enhanced with special tokens for handling image inputs:

- `<IMAGE>`: Marks the start of image content
- `<Image_Token>`: Represents individual image patch embeddings
- `<IMAGE_END>`: Marks the end of image content

### Vision-Language Adaptation

To bridge the semantic gap between visual and textual modalities, we implement:

- An MLP-based adapter that projects CLIP's visual features into the LLM's embedding space
- A strategy that treats visual features as special tokens in the language model:
  - Image patch embeddings are inserted at positions marked by `<Image_Token>`
  - The adapter ensures dimensional compatibility with the LLM's token embeddings
  - During generation, the model can attend to both text and image tokens seamlessly
  - The KV cache mechanism is modified to handle the hybrid input sequence efficiently

This architecture allows the model to:

1. Process images with high spatial fidelity
2. Maintain the powerful language understanding capabilities of Llama
3. Generate text that is grounded in both visual and textual context

Here's a nice figure that shows our model architecture from the [Emu 2](https://arxiv.org/abs/2312.13286) paper. Note that we did not train image generation feature in our model.

![image](assets/emu.png)

## Training

### Hyperparameters Pretraining

| Hyperparameters | Pretraining |
|-----------|--------|
| Learning rate | 1e-5 |
| LR decay | Cosine |
| Weight decay | 0 |
| Per device batch size | 8 |
| Gradient accumulation steps | 8 |
| Precision | BF16 |
| Total samples | 1.012M |
| Optimizer | AdamW BNB 8bit |
| Number of GPUs | 1 |
| Global batch size | 64 |

### Training Stats

| Stats | Pretraining |
|-----------|--------|
| GPU | 1xNvidia RTX 3090 |
| Training Time | 30 Hours |
| Visual Encoder Init. | openai/clip-vit-large-patch14-336 |
| Multi-Modal Modeling Init. | meta-llama/Llama-3.2-1B-Instruct |

## Results

### Pretraining Stage (Model 1)
