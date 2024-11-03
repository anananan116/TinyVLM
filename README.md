# TinyVLM

## Data Exploration and Initial Preprocessing

### Data Exploration

##### Data

- [Images used for training with descriptions](https://huggingface.co/datasets/BAAI/CapsFusion-120M)
- [Instruction Tuning #1](https://textvqa.org/dataset/)
- [Instruction Tuning #2](https://visualqa.org/download.html)

General Info on our Data:

- Our dataset provides over 13 million image links, but we are scaling down. We downloaded the first 5 million rows of the dataset, and of these we will only use the rows where the image link gives a successful response code.
- All initial images are not uniform in any regard, however during preprocessing, all images will be cropped
- 3 different descriptions for each images as features

### Preprocessing Steps

For preprocessing, we plan on doing the following:

- Downloading only images in which gives a successful response code (rows in the dataset corresponding to images without a successful response code will be disregarded)
- Cropping all the images to a desired 128 x 128 dimension
- Normalization is likely not needed, however will perform when needed
- Classification of the data, classifying each of the features
- Encorporate image descriptions to the desired images for training
- Prepare questions and answers for images to do instruction Tuning to the LLM pre-train model

Our dataset consists of images with a wide variety of aspect ratios. Some images are already square or nearly square, whereas others have extreme aspect ratios (very narrow/wide). To account for this, we will set an aspect ratio threshold of 0.6, where aspect ratio is defined as the minimum of the width and height over the maximum of the width and height. For images with an aspect ratio greater than or equal to 0.6, we will center crop the image, and images with an aspect ratio less than 0.6 will be padded. An example of our preprocessing for a single image is as follows: Say we have a very narrow image with a height of 400px and a width of 100px. The image will be padded to make it square, meaning black bars will be added to the left and right of the image, each one having a height of 400px and a width of 150px. The image will then be downscaled to 128x128.

### Data preparing for instruction tuning

1. Download questions and answers from the dataset
2. Create DataFrame from json data files which include questions and answers
3. Combine questions and answers and create a new DataFrame according to the image_id and answer_id
4. Create answers into a complete sentences
5. Add tags to indicate system prompt, user questions, and answers from the model
6. Combine system prompt, user questions, and answers into one col and label them with corrsponding image_id
7. Output the data as csv file
