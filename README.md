# Brain-Spine Neural Network  
Reflex Action Inspired Neural Network for Faster Inference

---

## Overview  

The **Brain Spine Neural Network** architecture is inspired by biological reflex actions. It combines two neural networks:  
- **Spine Network (Small, Fast Network):** A lightweight, fast-inference network providing rapid but slightly less detailed outputs.  
- **Brain Network (Large, Accurate Network):** A larger, more accurate network that performs detailed analysis at the cost of increased latency.  

The output of the Spine Network is given to the hidden layers of the Brain network which increases its accuracy.

This architecture allows for efficient processing by leveraging the spine for immediate, approximate results while utilizing the brain for high-precision outputs. 

<img width="800" alt="Brain Spine Neural Network Architecture" src="https://github.com/user-attachments/assets/42b5d5a1-73ac-47f8-bb39-bd95dfa8af9c">



## Project Structure

- documentation.md: provides detailed description regarding the project

- proving_hypothesis: this directory has file proving our assumption that adding an input of target variable which is 80% accurate in hidden layers will increase accuracy of the network

- data_preprocessing: this directory has the original dataset "obesity.csv", which is preprocessed ans saved as "final_data.csv". the file "split_data.py" divides this data into training and testing sets

- brain_spine_nn.py: this script trains and tests the final architechture of brain spine neural network which provided highest accuracy


### How To Run

to replicate results, follow the steps:

1. run the script data_preprocessing/split_data.py
2. run the script brain_spine_nn.py
3. The model weights are stored in spine_network.h5 and brain_network.h5


## Results
Best results were achieved when the activation functions were different in Brain and Spine Networks so that the patterns detected by them did not overlap and lead to overfitting. It was proved that integrating output of a smaller network into hidden layers of a larger network increased accuracy.  

