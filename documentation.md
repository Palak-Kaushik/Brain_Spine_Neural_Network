```markdown
# Brain-Spine Neural Network  
Reflex Action Inspired Neural Network for Faster Inference

---

## Overview  

The **Brain Spine Neural Network** architecture is inspired by biological reflex actions. It combines two neural networks:  
- **Spine Network (Small, Fast Network):** A lightweight, fast-inference network providing rapid but slightly less detailed outputs.  
- **Brain Network (Large, Accurate Network):** A larger, more accurate network that performs detailed analysis at the cost of increased latency.  

The output of the Spine Network is given to the hidden layers of the Brain network which increases its accuracy.

This architecture allows for efficient processing by leveraging the spine for immediate, approximate results while utilizing the brain for high-precision outputs.  

### Bio-inspired elements

This network takes inspiration from **reflex actions** in biological systems, which involve two key components: the **spinal cord** for rapid responses and the **brain** for detailed processing.  

1. **Spinal Cord Analogy (Spine Network):**  
   - Reflex actions are fast, automatic responses triggered by the spinal cord without involving the brain initially, ensuring quick reaction times.  
   - Similarly, the Spine Network is lightweight and processes input rapidly, providing approximate but immediate outputs for time-critical scenarios.  

2. **Brain Analogy (Brain Network):**  
   - The brain is responsible for detailed, conscious processing that requires more time but delivers greater accuracy and understanding.  
   - In this architecture, the Brain Network mirrors the brain’s role by performing deeper and more precise analysis, albeit at a higher computational cost.  

3. **Integration of Responses:**  
   - In reflex actions, the spinal cord’s initial response is complemented by the brain’s later, detailed analysis, creating a balance between speed and accuracy.  
   - Similarly, this neural network integrates the Spine's fast output into the Brain Network, combining the strengths of both for optimal performance.  

This bio-inspired design ensures that the system reacts quickly to immediate needs while still providing the accuracy and depth required for complex tasks.

---

## Benefits and Impact  

Advantages of the Brain Spine Neural Network:

- Speed-Accuracy Tradeoff: The small, fast network can be used for applications where speed is more critical than precision (e.g., real-time decision-making). The large, accurate network can be used for tasks requiring higher precision at the cost of increased computation time.

- Efficiency: By leveraging the small network, the system can handle tasks requiring lower precision with reduced energy consumption and computational cost. The larger network is reserved for cases where higher accuracy is needed, optimizing resource utilization.

- Parallel processing: The spine and Brain networks can be optimised to execute parallely.

The Brain Spine Neural Network is highly suitable for time-sensitive applications, such as:  
- **Autonomous Vehicles:** The spine network can detect the presence of an object quickly, while the brain network performs detailed classification and analysis of the object.  
- **Real-Time Decision Systems:** In scenarios where quick preliminary decisions are essential, followed by detailed reasoning when computational time permits.  

This dual-network architecture ensures an optimal tradeoff between speed and accuracy, improving both efficiency and adaptability.  

---

## Methodology  

### Dataset  
This project uses the **Obesity Prediction Dataset**:  
**Palechor, F.M., & Manotas, A.D. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru, and Mexico. _Data in Brief, 25._**  

#### Data Preprocessing  


1. Normalization and Encoding was performed on the numerical and categorical attributes respectively.

2. A "Binary" column was added, representing whether an individual has a healthy weight (`0`) or unhealthy weight (`1`).  
   - **Spine NN:** Predicts the binary attribute (healthy/unhealthy).  
   - **Brain NN:** Performs multi-class classification for detailed obesity levels.  


#### Data Splitting  
The script `split_data.py` creates training and testing splits and saves them in the `split_data` directory.

---

### Experimental Design  

**Hypothesis:** We assumed that passing the Spine Network's output as an input to the Brain Network improves the accuracy of the Brain, even if the spine gives at least 80% correct predictions. This has been proved by experiments conducted, which are present in "Proving Hypothesis" directory. 

#### Experiments Conducted  

Two versions were tested for comparison: one where the Spine Network's output was passed to the Brain Network and another where it was not.

Initially, both networks used ReLU activation in their hidden layers. In this setup, the Brain Network's accuracy was subpar and performed better when the Spine's output was not integrated.

Subsequently, the Brain Network's activation functions were changed to Leaky ReLU and Tanh, so that the patterns detected by Brain and Spine networks did not overlap and thus increase accuracy. This modification improved the Brain Network's accuracy when the Spine's output was integrated.

The Spine's output was tested by passing it at different points within the Brain Network—first hidden layers, middle layers, and last layers. The best performance was achieved when the Spine's output was integrated into the middle layers of the Brain Network.


---

### Results  

Best results were achieved when the activation functions were different in Brain and Spine Networks so that the patterns detected by them did not overlap and lead to overfitting. It was proved that integrating output of a smaller network into hidden layers of a larger network increased accuracy.  


---

## Background research