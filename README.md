# TinyVLM

## Data Exploration and Initial Preprocessing

### Data Exploration

##### Data
- [Images used for training with descriptions](https://huggingface.co/datasets/BAAI/CapsFusion-120M)
- [Instruction Tuning #1](https://textvqa.org/dataset/)
- [Instruction Tuning #2](https://visualqa.org/download.html)

General Info on our Data:

- Over 13 million image links avaliable, but we are scaling down and using the first 10,000
- All initial images are not uniform in any regard, however during preprocessing, all images will be cropped
- 3 different discriptions for each images as features

### Preprocessing Steps

For preprocessing, we plan on doing the following:

- Downloading only images in which gives a successful response code
- Cropping all the images to a desired 128 x 128 dimension
- Normalization is likely not needed, however will perform when needed
- Classification of the data, classifying each of the features
- Encorporate image descriptions to the desired images for training
- Prepare questions and answers for images to do instruction Tuning to the LLM pre-train model

### Data preparing for instruction tuning
1. download questions and answers form the dataset
2. Create data frame from json data files which include questions and answers
3. Combine question and answers and create a new dataframe according to the image_id and answer_id
4. Create answers into a complete sentences
5. Add tags to indicate system prompt, user questions, and answers from the model
6. Combine system prompt, user questions, and answers into one col and label them with corrsponding image_id
7. Output the data as csv file
