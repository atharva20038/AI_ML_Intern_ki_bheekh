### Fine Tuning LLM for Psychologist Therapy
The aim of this project is to fine tune the LLAMA2 large language model on a therapy dataset to fine tune a model specifically for talking to people in distress.

### Dataset 
The dataset for this project is the counsel-chat dataset which contains 2.7k samples of Question-Answer pairs. I have used this dataset to improve the model performance and then benchmark the model performance to 
get a better understanding of its performance on the dataset. 
https://huggingface.co/datasets/nbertagnolli/counsel-chat

### Running the pipeline 
The jupyter notebook in the repository runs the pipeline for fine-tuning the model. 

The LLAMA model is loaded in 4bits using bitsandbytes library.
The dependencies needed for the model are added in the jupyter notebook. 

### Compute Resources Used 
The jupyter notebook was hosted on kaggle with 2 GPUs each having a limit of 15GB and the model is loaded on the GPU. 
