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