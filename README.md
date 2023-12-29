### Fine Tuning LLM for Psychologist Therapy
The aim of this project is to fine tune the LLAMA2 large language model on a therapy dataset to fine tune a model specifically for talking to people in distress.

### Dataset 
The dataset for this project is the counsel-chat dataset which contains 2.7k samples of Question-Answer pairs. I have used this dataset to improve the model performance and then benchmark the model performance to 
get a better understanding of its performance on the dataset. 
https://huggingface.co/datasets/nbertagnolli/counsel-chat

### Running the pipeline 
The jupyter notebook in the repository runs the pipeline for fine-tuning the model. 

The LLAMA model is loaded in 4bits using bitsandbytes library for quantizating the model. 
The dependencies needed for the model are added in the jupyter notebook. 
The optimiser used is 8-bit adam and the model is fine-tuned initially on 100 Q&A pairs initially.

### Compute Resources Used 
The jupyter notebook was hosted on kaggle with 2 GPUs each having a limit of 15GB and the model is loaded on the GPU. 


###### Increasing context length using word embeddings

I also researched about different methods to increase the context length of mpt-7b model.
During the process, I understood that the model contains several block with each block containing MultiHeadAttention & Linear layers.

1. ALiBi approach, which is built into the model. The model does not utilise postional encodings.
2. xPos: It is the approach which extends the positional embedding by setting the base number in the sin/cos function as an output of a function paramterised by L(Length of sequence).
3. Linear Scaling & Positional Interpolation : The approach multiplies and divides by L/L* which is the old maximum training length/new_length of the sequence.
4. RoPe : It is the technique for Rotation Positional Encoding which uses positional encoding along with a rotational transformation.
   The rotational matrix is multiplied by the Query & Key Matrices.
   
