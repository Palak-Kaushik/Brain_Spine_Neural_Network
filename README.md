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

- `documentation.md`: provides detailed description regarding the project

- `proving_hypothesis`: this directory has file proving our assumption that adding an input of target variable which is 80% accurate in hidden layers will increase accuracy of the network

- `data_preprocessing`: this directory has the original dataset `obesity.csv`, which is preprocessed ans saved as `final_data.csv`. the file `split_data.py` divides this data into training and testing sets

- `brain_spine_nn.py`: this script trains and tests the final architechture of brain spine neural network which provided highest accuracy


### How To Run  

To replicate the results, follow these steps:  

1. **Run the data preprocessing script:**  
   ```bash
   python data_preprocessing/split_data.py
   ```  

2. **Run the main neural network script:**  
   ```bash
   python brain_spine_nn.py
   ```  

3. **The trained model weights will be saved as:**    
     - `spine_network.h5`  
     - `brain_network.h5`  


## Results
Best results were achieved when the activation functions were different in Brain and Spine Networks so that the patterns detected by them did not overlap and lead to overfitting. It was proved that integrating output of a smaller network into hidden layers of a larger network increased accuracy.  

| Brain-Spine output connection | Layer Activation Functions | Accuracy Achieved in Brain Network |
|-----------------------|-------------------|-------------------|
| No                    | Similar           | 66.2%             |
| Yes                   | Similar           | 62%               |
| No                    | Different         | 88.6%             |
| **Yes**                   | **Different**         | **90.4%**             |

This data compares the accuracy of the Brain Network based on the activation functions used in both the Spine and Brain Networks. Accuracy of Spine network was constant at 86%.

Layer Activations: Refers to whether the activation functions used in both networks (Spine and Brain) are the same or different.
Accuracy Achieved: Shows the performance of the Brain Network when integrating the Spine’s output.
Results:

- No Spine connection, Similar Activations (66.2%): Accuracy is higher than when spine connection is present, as no overfitting is taking place.
- Spine connection present, Similar Activations (62%): When the Spine’s output is passed to the Brain Network, but both networks use similar activations, accuracy decreases even more, due to pattern overlap causing redundant learning.
- No spine connection, Different Activations (88.6%): When different activation functions are used in both networks, the Brain Network can process unique patterns from the Spine Network, improving accuracy.
- spine connection present, Different Activations (90.4%): The best performance occurs when both networks use different activation functions, and the Spine’s output is integrated into the Brain's middle layers, yielding the highest accuracy.

In short, different activation functions between the networks prevent pattern overlap, reduce overfitting, and significantly enhance the Brain Network's accuracy.
