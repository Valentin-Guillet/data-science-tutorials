# Reinforcement Learning: Solving MinAtar games using Deep Q-Learning

**Difficulty**: medium

## Overview

This project aims to explore and implement Deep Q-learning algorithms in reinforcement learning, specifically focusing on MinAtar games. The project will encompass the development of a Deep Q-Learning agent from scratch, training and testing it in different environments, and optionally, comparing its performance with a Munchausen agent.

## Objectives

1. Implement a Deep Q-Learning Agent:
  - Develop a Deep Q-Learning agent using PyTorch, guided by resources like the PyTorch Deep Q-Learning Tutorial (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
  - Ensure thorough code documentation and testing.

2. Training and evaluating on the CartPole Environment
  - Test the convergence of the agent in the CartPole environment (reach 200 rewards).
  - Record the learning progress and analyze the performance metrics.

3. Training in Additional Environments:
  - Train the agent in at least two environments, including MinAtar, as outlined in the MinAtar Environment GitHub Repository: https://github.com/kenjyoung/MinAtar.
  - Generate GIFs to demonstrate the trained agent's performance visually.

4. Optional - Munchausen Agent Implementation:
  - Implement and train a Munchausen agent, as detailed in the Munchausen Q-Learning Paper (https://arxiv.org/abs/2007.14430) and its implementation at https://github.com/BY571/Munchausen-RL.
  - Compare its performance against the standard DQN agent.

5. Documentation:
  - Create a comprehensive README file highlighting the work, methodology, and results.
  - Develop a separate REPORT.md file explaining the mathematical foundations and workings of Deep Q-Learning and, optionally, Munchausen Q-Learning. Use resources like the Deep Q-Learning Paper: https://arxiv.org/abs/1312.5602 and Deep Q-Learning Paper Explained: https://www.youtube.com/watch?v=nOBm4aYEYR4  for reference.

## Expected Deliverables

1. Codebase: Complete, well-documented Python code for the Deep Q-Learning agent and, if pursued, the Munchausen agent.
2. GIFs and videos: Animated visuals displaying the trained agents in action in different environments.
3. Documentation:
  - `README.md`: A well-crafted document highlighting the project’s purpose, structure, and results.
  - `REPORT.md`: A detailed explanatory document focusing on the theoretical aspects of Deep Q-Learning and Munchausen Q-Learning.
4. Comparative Analysis: (Optional) A comparative study of the Deep Q-Learning and Munchausen agents, if implemented.

## Resources and References

- PyTorch Tutorial on Deep Q-Learning: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- MinAtar Environment GitHub: https://github.com/kenjyoung/MinAtar
- Deep Q-Learning Paper: https://arxiv.org/abs/1312.5602
- Deep Q-Learning Paper Explained: https://www.youtube.com/watch?v=nOBm4aYEYR4
- CleanRL for Q-Learning Implementation: https://github.com/vwxyzjn/cleanrl
- Munchausen Q-Learning Paper: https://arxiv.org/abs/2007.14430
- Munchausen RL Implementation: https://github.com/BY571/Munchausen-RL


# Reinforcement Learning: Implementing PPO in Vectorized Robotic Environments

**Difficulty**: hard

## Overview

This project delves into implementing and analyzing the Proximal Policy Optimization (PPO) algorithm in vectorized robotic environments, such as those provided by Mujoco and BRAX. The primary goal is to develop a PPO agent from scratch and rigorously evaluate its performance in various complex, high-dimensional environments.

## Objectives

1. Implement the PPO Algorithm:
  - Develop the PPO algorithm using PyTorch, drawing guidance from resources like CleanRL's PPO Implementation: https://github.com/vwxyzjn/cleanrl/tree/master.
  - Ensure the code is well-documented and tested for robustness.

2. Initial Testing and Validation in Inverted Pendulum:
  - Initially test the PPO algorithm in a simpler Inverted Pendulum environment to validate its functionality.
  - Analyze the performance metrics and stability of the agent.

3. Evaluation in Vectorized Environments:
  - Employ the BRAX framework, as referenced in Jax Vectorized Environments (https://github.com/google/brax), to test the PPO agent in more complex and highly vectorized environments like Ant or Humanoid.
  - Generate GIFs to visually demonstrate the trained agent's performance.

4. Optional - Implement PPO EWMA:
  - Optionally implement PPO EWMA (Exponentially Weighted Moving Average), as detailed in the PPO EWMA Paper (https://arxiv.org/abs/2110.00641) and its codebase (https://github.com/openai/ppo-ewma).
  - Conduct a comparative analysis of standard PPO and PPO EWMA.

5. Documentation:
  - Develop a comprehensive README file that showcases the project’s purpose, methodology, and key findings.
  - Create a `REPORT.md` file detailing the theoretical aspects and mathematical foundations of PPO (and optionally, PPO EWMA).

## Expected Deliverables

1. Codebase: Complete, well-documented Python code implementing the PPO algorithm and, if pursued, PPO EWMA.
2. Performance Analysis: Detailed analysis of the agent's performance in various environments, including comparative studies if PPO EWMA is implemented.
3. GIFs and Videos: Animated visuals displaying the trained agents in action.
4. Documentation:
  - `README.md`: A descriptive document highlighting the project’s objectives, structure, and outcomes.
  - `REPORT.md`: A technical report explaining the workings of PPO and PPO EWMA, incorporating mathematical explanations.

## Resources and References

- CleanRL's PPO Implementation in PyTorch: https://github.com/vwxyzjn/cleanrl/tree/master
- Jax Vectorized Environments (BRAX): https://github.com/google/brax
- PPO Paper: https://arxiv.org/abs/1707.06347
- PPO Explained (Video): https://www.youtube.com/watch?v=5P7I-xPq8u8
- Live Coding PPO (YouTube Video): https://www.youtube.com/watch?v=hlv79rcHws0
- PPO EWMA Paper: https://arxiv.org/abs/2110.00641
- PPO EWMA GitHub Repository: https://github.com/openai/ppo-ewma


# Reinforcement Learning: RLHF for Enhancing Large Language Models

**Difficulty**: medium

## Overview

This project explores the integration of **Human Feedback (HF)** into the training of Large Language Models (LLMs) using **Reinforcement Learning (RL)**, specifically the RLHF pipeline. The aim is to improve Transformer-based models like GPT or LLaMA by incorporating human preferences to produce more aligned, accurate, and contextually relevant outputs.

**Disclaimer:** Full-scale RLHF training is computationally intensive. The goal of this project is to implement and demonstrate a **functional training pipeline** for the RLHF stages, rather than achieving state-of-the-art results.

## Objectives

1. Understanding and Utilizing Causal Transformers:
  - Explain the workings of the Transformer architecture in the REPORT.md, focusing on the components (Attention and MLP) and the mathematical foundation.
  - Detail the concept of **Causal Transformers** (e.g., GPT, LLaMA) and their self-supervised training methodology (Causal Language Modeling).

2. Implementing the RLHF Pipeline:
  - Describe in REPORT.md the operation of Reinforcement Learning with Human Feedback in the context of LLMs (Reward Model training and PPO optimization).
  - Utilize existing high-level libraries (e.g., Hugging Face TRL) for the practical implementation to manage complexity.

3. Training a Reward Model:
  - Train a Reward Model using a pre-trained LLM (e.g., $\text{GPT}-2$ or similar size model), referencing the Reward Modeling Example from TRL (https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py).
  - Document the training process, data utilized, and analysis of the resulting reward scores.

4. Optimization with Proximal Policy Optimization (PPO):
  - Implement PPO for the final optimization of the LLM using the trained Reward Model.
  - The Hugging Face TRL Quickstart Guide (https://huggingface.co/docs/trl/quickstart#minimal-example) serves as the core resource for this implementation.
  - Generate and compare sample text outputs from the base model and the RLHF-optimized model to demonstrate the effect of the optimization.

5. Documentation:
  - Create a detailed README.md file that outlines the project's goals, methodology, and significant findings.
  - The REPORT.md should provide an in-depth explanation of Transformer architectures, causal transformers, and the application of RL with human feedback in LLMs.

## Expected Deliverables

1. Codebase: Well-documented Python code for the entire RLHF pipeline, including Reward Model training and PPO optimization scripts.
2. Sample Outputs: Generated text samples from the optimized LLM (including a comparison between base and RLHF output).
3. Documentation:
  - `README.md`: An overview document highlighting the project’s objectives, structure, and key results.
  - `REPORT.md`: A technical report detailing the theoretical aspects of Transformers and the application of RLHF.

## Resources and References

- Hugging Face TRL for Reward Modeling: https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py
- Hugging Face TRL Quickstart Guide (Core Implementation Resource): https://huggingface.co/docs/trl/quickstart#minimal-example
- OpenAI's RLHF Papers for ChatGPT:
  - First RLHF Paper: https://proceedings.neurips.cc/paper_files/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf
  - GPT-3.5/ChatGPT Paper: https://arxiv.org/abs/2203.02155
- Anthropic AI's Work on RLHF: https://arxiv.org/abs/2204.05862
- Blog post for RLHF with PPO: https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo
- RLHF Explained (Blog): https://huggingface.co/blog/rlhf
- PPO Paper: https://arxiv.org/abs/1707.06347
- CleanRL's PPO Implementation in PyTorch (for understanding PPO): https://github.com/vwxyzjn/cleanrl/tree/master


# Optimizing Large Language Model Inference with 8-bit Quantization

**Difficulty**: medium

## Overview

This project aims to optimize the inference process of **Large Language Models (LLMs)**, particularly the LLaMA family, by implementing and analyzing 8-bit quantization techniques. The primary focus is on reducing the memory footprint and computational requirements of these large models while rigorously maintaining performance for text generation tasks.

## Objectives

1. Understanding Causal Transformer Architectures:
  - Provide a detailed explanation of how Transformer and Causal Transformer models (like GPT and LLaMA) function in the REPORT.md.
  - Focus on the internal architecture of LLaMA models, utilizing resources like the LLaMA Paper (https://arxiv.org/abs/2302.13971) and LLaMA 2 Paper (https://arxiv.org/abs/2307.09288).

2. Implementing 8-bit Quantization:
  - Explore the concept of representing neural network weights in 8-bit integers instead of the standard 32-bit floating point precision, referencing the 8-bit LLM Paper (https://arxiv.org/abs/2208.07339).
  - Convert a small LLaMA model to an 8-bit representation using tools like Bitsandbytes (https://github.com/timdettmers/bitsandbytes).

3. Performance Assessment:
  - Implement text generation tasks and assess the performance, latency, and memory usage of the 8-bit quantized model compared to its original 32-bit counterpart.

4. Optional - Flash Attention Mechanism:
  - Optionally, research and explain the Flash Attention v1 and v2 mechanisms, understanding their role in improving LLM efficiency by reducing HBM (High Bandwidth Memory) access.

5. Documentation:
  - Create a comprehensive README.md file that outlines the project's purpose, methodology, and key findings.
  - The REPORT.md should detail LLaMA architecture, the 8-bit quantization process, and the impact on model performance and resource consumption.

## Expected Deliverables

1. Codebase: Well-documented Python code for loading the base LLaMA model, applying 8-bit quantization, and generating text samples.
2. Performance Analysis: Detailed quantitative analysis of the memory footprint, inference speed, and output quality of the quantized model.
3. Documentation:
  - `README.md`: A document highlighting the project’s objectives, methodology, and key results.
  - `REPORT.md`: A technical report detailing the theoretical aspects of Transformers, LLaMA models, and 8-bit quantization.

## Resources and References

- 8-bit LLM Paper: https://arxiv.org/abs/2208.07339
- LLaMA Paper: https://arxiv.org/abs/2302.13971
- LLaMA 2 Paper: https://arxiv.org/abs/2307.09288
- Bitsandbytes GitHub Repository: https://github.com/timdettmers/bitsandbytes
- Flash Attention v1 Paper: https://arxiv.org/abs/2205.14135
- Flash Attention v2 Paper: https://arxiv.org/abs/2307.08691
- Hugging Face's Flash Attention Documentation: https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention
- Flash Attention Explained (Blog Post): https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad


# Generative Models: Variational Autoencoder on CelebA with ELBO

**Difficulty**: easy

## Overview

This project focuses on the implementation and theoretical understanding of the **Variational Autoencoder (VAE)**, a powerful generative model. The primary goal is to implement a VAE on the CelebA face dataset, with a specific emphasis on the **Evidence Lower Bound (ELBO)**, which serves as the core objective function for VAE training.

## Objectives

1. Understanding and Explaining VAEs:
  - Explore the basic concepts, architecture, and mechanism of Variational Autoencoders, detailing how they differ from traditional Autoencoders.
  - Provide a clear, technical explanation of VAEs in the REPORT.md.

2. Dataset Preparation:
  - Utilize the CelebA dataset (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for training.
  - Implement necessary data loading and preprocessing steps (e.g., resizing, normalization) suitable for image input to the VAE model.

3. Implementing the VAE:
  - Implement the VAE model using PyTorch, including both the Encoder (to parameterize the latent distribution) and the Decoder (to reconstruct the image).

4. Focusing on ELBO (Evidence Lower Bound):
  - Explain the concept and mathematical derivation of the ELBO in the REPORT.md.
  - Demonstrate how ELBO is broken down into the **Reconstruction Loss** and the **KL Divergence Loss**, and how it is used as the objective function during VAE training.

5. Training and Results Analysis:
  - Train the VAE model on the CelebA dataset, ensuring the ELBO is correctly implemented in the loss function.
  - Analyze the results by focusing on the quality of generated images and examining the relationship between the two ELBO components during the training process.

6. Optional - Implementation of VQ-VAE:
  - Research and document the **Vector Quantized VAE (VQ-VAE)** architecture (https://arxiv.org/abs/1711.00937).
  - Optionally, implement the VQ-VAE and compare its generative capabilities against the standard VAE.

## Expected Deliverables

1. Codebase: Complete, well-documented Python code for the VAE model, including data preprocessing, model implementation, and training scripts.
2. Generated Samples: Visual outputs showing the model's ability to generate new, realistic CelebA-style images.
3. Performance Analysis: An analysis focusing on the role of ELBO in training convergence and the qualitative assessment of image generation.
4. Documentation:
  - `README.md`: A summary of the project’s objectives, methodology, and outcomes.
  - `REPORT.md`: A detailed report explaining the theoretical aspects of VAEs, the mathematical foundation of ELBO, and practical applications.

## Resources and References

- CelebA Dataset: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- VAE Original Paper: https://arxiv.org/abs/1312.6114
- VQ-VAE Paper: https://arxiv.org/abs/1711.00937
- PyTorch VAE Implementation Examples: https://github.com/AntixK/PyTorch-VAE
- Blog Post on VAE Theory: https://avandekleut.github.io/vae/


# Understanding and Implementing Diffusion Models for Image Generation on CelebA

**Difficulty**: hard

## Overview

This project provides an in-depth study and practical implementation of **Diffusion Models** for high-quality image synthesis, focusing on the CelebA face dataset. The core technical challenge involves designing and integrating the **U-Net architecture** as the noise prediction network, which is central to the iterative denoising process in Diffusion Models.

## Objectives

1. **Theoretical Foundation of Generative Models:**
  * **Diffusion Models:** Comprehensively explore the two core processes: the **Forward Diffusion Process** and the learned **Reverse Diffusion Process**.
  * **U-Net Architecture:** Analyze the original U-Net design (from image segmentation) and its adaptation as the $\epsilon$-prediction network in Denoising Diffusion Probabilistic Models (DDPM). Explain how the **symmetric encoder-decoder structure** and **skip connections**  are crucial for preserving fine spatial details during the denoising steps.

2. **Dataset Preparation:**
  * Utilize the CelebA dataset. Implement a robust data pipeline to handle image loading, resizing (e.g., to $64 \times 64$ or $128 \times 128$), normalization (e.g., to the $[-1, 1]$ range), and efficient batch loading.

3. **Implementing the DDPM Framework:**
  * **Noise Schedule:** Define and implement the variance schedule ($\beta_t$ and $\alpha_t$) for the forward process.
  * **U-Net Integration:** Implement the U-Net model. This network must take two inputs: the noisy image ($x_t$) and the **timestep ($t$)**. The timestep must be encoded (e.g., using sinusoidal position embeddings) and integrated into the convolutional blocks.
  * **Training Objective:** Implement the simplified DDPM loss function, where the U-Net is trained to predict the noise $\epsilon$ that was added to the image at time $t$.

4. **Image Generation and Sampling:**
  * Implement the iterative reverse (sampling) process. Start with pure Gaussian noise ($x_T$) and use the trained U-Net to progressively estimate and remove noise until a clean image ($x_0$) is generated.

5. **Documentation and Analysis:**
  * Analyze the generated image quality (fidelity and diversity).
  * Document the architectural choices (e.g., number of layers, attention mechanisms) in the U-Net and discuss their impact on image quality.

## Expected Deliverables

1. **Codebase:** Complete, runnable Python code for the DDPM implementation, including the custom U-Net model, data loading, and training loop (e.g., using PyTorch).
2. **Generated Images:** A visualization of generated images from the CelebA dataset, demonstrating the model's capacity for realistic synthesis.
3. **Documentation:**
  * `README.md`: Project summary, methodology, and how to run the code.
  * `REPORT.md`: A comprehensive report detailing the mathematics of the forward and reverse diffusion processes, the structural components of the U-Net, and an analysis of the training convergence and generative results.

## Resources and References

-   U-Net Paper: https://arxiv.org/abs/1505.04597
-   Tutorial on Diffusion Model: https://github.com/d9w/gen_models/blob/main/Score_Based_Generative_Modeling.ipynb
-   Score-Based Generative Modeling through Stochastic Differential Equations: https://arxiv.org/abs/2011.13456
-   Denoising Diffusion Probabilistic Models: https://arxiv.org/abs/2006.11239
-   You can check out [What is U-Net in Diffusion Models?]($2.2$) to get a quick visual overview of how the U-Net architecture is used within the diffusion framework.


# Fine-Tuning Diffusion Models with LoRA

**Difficulty**: medium

## Overview

This project explores the concept of **Low-Rank Adaptation (LoRA)** as an efficient method to fine-tune pre-trained **Diffusion Models**. The main goal is to understand LoRA's mechanics, implement the technique, and apply it to enhance a diffusion model for a specific image or text generation task using a  dataset of your choice.

## Objectives

1. **Understand and Document LoRA:**
  * Thoroughly research the principles and mathematical foundations of Low-Rank Adaptation (LoRA).
  * Document this understanding in the `REPORT.md` file, focusing on its application in fine-tuning neural networks.

2. **Dataset Selection and Preparation:**
  * Select and prepare a suitable image or text dataset for fine-tuning a diffusion model to a specific task.

3. **Implement LoRA Fine-Tuning:**
  * Implement a diffusion model and apply the LoRA technique for efficient fine-tuning, leveraging resources like the Hugging Face's LoRA Documentation.
  * Experiment with multiple LoRA configurations (e.g., different ranks) to test the versatility of the method.

4. **Output Generation and Comparison:**
  * Use the LoRA-fine-tuned model to generate outputs on the selected dataset.
  * Evaluate and compare the quality of these outputs against those generated by the base (un-fine-tuned) model.

5. **Analysis and Discussion:**
  * Analyze the effectiveness of LoRA in reducing training costs and improving performance for the target task.
  * Discuss the changes observed in the generated outputs across different LoRA configurations in the `REPORT.md`.

6. **Documentation:**
  * Develop a comprehensive `README.md` and a detailed `REPORT.md` that includes the theoretical background and practical findings.

## Expected Deliverables

1. **Codebase:** Complete, well-documented Python code for implementing LoRA with a diffusion model and generating outputs.
2. **Generated Outputs:** A collection of visuals or text outputs showcasing the effects of LoRA fine-tuning and a comparison against the base model.
3. **Documentation:**
  * `README.md`: A document summarizing the project’s objectives, methodology, and key findings.
  * `REPORT.md`: A detailed report explaining the theoretical aspects of Diffusion Models and LoRA, along with the results of the practical implementation and analysis.

## Resources and References

* LoRA Paper: https://arxiv.org/abs/2106.09685
* Explanatory Video on LoRA: https://www.youtube.com/watch?v=vjEPXSCbmDE
* Hugging Face's LoRA Documentation: https://huggingface.co/docs/diffusers/training/lora#text-to-image


# Group Equivariant Neural Networks: Implementing Rotation Invariance

**Difficulty**: hard

## Overview

This project explores the mathematical principles and practical implementation of **Group Equivariant Neural Networks (G-CNNs)**, with a focus on achieving **rotation invariance**. This property is essential for tasks involving data where orientation is arbitrary or unpredictable, such as remote sensing (e.g., satellite imagery like EuroSAT) or microscopy data. The project involves implementing and testing specific group convolutions like $C_4$ (Cyclic Group) and general group convolutions.

## Objectives

1. **Grasp and Document Group Theory Fundamentals:**
  * Explain the mathematical definition and properties of a **group**.
  * Explain how standard convolution achieves **translation equivariance** (and invariance) through mathematical explanation and a concise code demonstration.

2. **Implement Cyclic Group Convolution ($C_4$):**
  * Develop and implement **Cyclic Group Convolution** for the $C_4$ group (90-degree rotations).
  * Test the $C_4$-equivariant network for classification tasks on the **EuroSAT dataset**.

3. **Implement General Group Convolutions:**
  * Implement a **general group convolution** framework (e.g., $SO(2)$ continuous rotations or a larger discrete group $C_n$).
  * Explain the key mathematical and implementation differences between cyclic group convolutions ($C_n$) and general group convolutions.
  * Apply and test the general group convolution on the **EuroSAT dataset**.

4. **Advanced Challenges (Optional):**
  * **[Bonus] Deep Convolutional GAN with G-CNNs:** Create and implement a Deep Convolutional GAN (DCGAN) using either the Cyclic or General Group Convolutions and compare the quality of generated images against a standard DCGAN.
  * **[Hardcore] Steerable Convolutions:** Research and implement Steerable Convolutions, which offer continuous rotation equivariance ($SO(2)$), and demonstrate their properties.

## Expected Deliverables

1. **Codebase:** Complete, well-documented Python code for the implementation of $C_4$ and general Group Convolutions. Include code for any optional advanced challenges completed.
2. **Documentation:**
  * `README.md`: A clear document summarizing the project's aims, methodology, and key experimental outcomes.
  * `REPORT.md`: An in-depth technical report elaborating on the mathematical foundations of group theory, equivariance, the specific implementation of the $C_4$ and general group convolutions, and a thorough analysis of the results on EuroSAT.

## Resources and References

* **EuroSAT Dataset:** https://github.com/phelber/EuroSAT
* **UvA - An Introduction to Group Equivariant Deep Learning:** https://uvagedl.github.io/
* **Tutorial on Group Equivariant Neural Networks:** https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.html
* **Group Equivariant Neural Network Research Paper:** https://arxiv.org/abs/1602.07576


# Emotion Recognition from Speech using Mel Spectrograms and CNNs

**Difficulty**: easy

## Overview

This project focuses on the implementation of a deep learning solution for **Emotion Recognition from Speech (ERS)**. The primary objective is to leverage the visual representation of sound, specifically **Mel Spectrograms**, and train **Convolutional Neural Networks (CNNs)** to classify various emotional states. The project will utilize the Acted Emotional Speech Dynamic Database for training and evaluation.

## Objectives

1. **Dataset Acquisition and Exploration:**
  * Download and explore the **Acted Emotional Speech Dynamic Database** from DagsHub (https://dagshub.com/kingabzpro/Acted-Emotional-Speech-Dynamic-Database).
  * Perform initial data analysis to understand the distribution of emotions and key audio characteristics within the dataset.

2. **Audio Preprocessing and Feature Extraction:**
  * Preprocess the raw audio files (e.g., normalization, alignment).
  * Convert the preprocessed audio clips into **Mel Spectrograms**, ensuring optimal parameters are chosen to best capture emotional features.

3. **CNN Model Design and Implementation:**
  * Develop and implement one or more **CNN architectures** (e.g., a standard 2D CNN) tailored for the classification of Mel Spectrogram images.
  * Design the model to effectively identify subtle time-frequency patterns that correlate with different emotional states.

4. **Model Training and Evaluation:**
  * Train the CNN model(s) using the Mel Spectrogram features.
  * Implement a robust validation strategy (e.g., cross-validation or a clear train/validation/test split) to evaluate the model's performance and generalization ability.

5. **Performance Optimization:**
  * Assess model performance using metrics such as **accuracy, precision, and recall** per class.
  * Experiment with different CNN hyperparameters, architectures, or data augmentation techniques to enhance classification accuracy.

6. **Documentation and Reporting:**
  * Thoroughly document the entire project workflow, from data handling to model evaluation.

## Expected Deliverables

1. **Codebase:** Complete, well-documented Python codebase for audio preprocessing, Mel Spectrogram conversion, CNN model development, training, and evaluation.
2. **Model Evaluation Reports:** Detailed analysis of the model's performance, including confusion matrices and metrics (accuracy, precision, recall) across all emotional classes.
3. **Comprehensive Documentation:**
  * `README.md`: A concise overview of the project, its objectives, methods, and key findings.
  * `REPORT.md`: An in-depth report outlining the complete workflow, including Mel Spectrogram generation parameters, CNN architecture details, results, and insights into challenges and potential improvements.

## Resources and References

* **Acted Emotional Speech Dynamic Database (DagsHub):** https://dagshub.com/kingabzpro/Acted-Emotional-Speech-Dynamic-Database


# Lipschitz Neural Networks: Enhancing Robustness via $1$-Lipschitz Constraints

**Difficulty**: hard

## Overview

This project focuses on the theoretical understanding and practical implementation of **Lipschitz Neural Networks**, specifically targeting a **$1$-Lipschitz constant** constraint on the network layers. The primary goal is to enhance model stability, robustness against adversarial examples, and potentially improve generalization. This will be achieved using techniques like **spectral norm clipping** within a classification task on the **CIFAR-10** dataset.

## Objectives

1. **Grasp and Document Lipschitz Theory:**
  * Explain the mathematical concept of a **Lipschitz constant** and its critical significance for stability and robustness in neural networks.
  * Detail the method for achieving the desired $1$-Lipschitz constant by clipping the **maximum eigenvalue (spectral norm)** of a layer's weight matrix.

2. **Implement Spectral Norm Lipschitz Constraint:**
  * Develop a Lipschitz-constrained neural network (inspired by libraries like Deel-TorchLip) by implementing **spectral norm clipping** to enforce the $1$-Lipschitz property on convolutional and/or fully connected layers.
  * Apply the implemented network to perform a classification task on the **CIFAR-10** dataset.

3. **Advanced Exploration (Optional):**
  * Implement a Lipschitz neural network using an alternative method, such as **Bjork orthogonalization**.
  * Compare and contrast the spectral norm clipping and Bjork orthogonalization methods in the `REPORT.md`, focusing on implementation complexity, computational cost, and performance impact.

4. **Performance Evaluation and Analysis:**
  * Evaluate the network's performance in terms of **classification accuracy** and its **robustness** (e.g., against simple adversarial attacks, if time permits).
  * Analyze and discuss the benefits and limitations of using Lipschitz constraints in the `REPORT.md`.

## Expected Deliverables

1. **Codebase:** Complete, well-documented Python code for implementing Lipschitz neural networks using spectral norm and, optionally, Bjork orthogonalization. Include the code for the CIFAR-10 classification task.
2. **Comprehensive Documentation:**
  * `README.md`: A concise overview of the project, its objectives, methods, and key findings.
  * `REPORT.md`: An in-depth report outlining the complete workflow, including Mel Spectrogram generation parameters, CNN architecture details, results, and insights into challenges and potential improvements.

## Resources and References

* **Implementation of Lipschitz Neural Networks (Deel-TorchLip):** https://github.com/deel-ai/deel-torchlip
* **Paper on Misconceptions about Lipschitz Neural Networks:** https://proceedings.neurips.cc/paper_files/paper/2022/file/7eb3d8ae592966543170a65e6b698828-Paper-Conference.pdf
* **Paper on 1-Lipschitz Neural Networks from an Optimal Transport Perspective:** https://hal.science/hal-03693355v2/file/OTNN_explainability.pdf
* **Research on Robust One-Class Classification with 1-Lipschitz Neural Networks:** https://arxiv.org/abs/2303.01978


# Graph Neural Networks (GNNs): Property Prediction on the ZINC Dataset

**Difficulty**: medium

## Overview

This project is an exploration of **Graph Neural Networks (GNNs)**, a specialized class of deep learning models designed to operate on non-Euclidean data structures (graphs). The project aims to cover the fundamental concepts of graph theory, GNN mechanics, and their application to a real-world problem: predicting molecular properties using the **ZINC chemical compounds dataset** via the **PyTorch Geometric** library.

## Objectives

1. **Grasp and Document Graph Fundamentals:**
  * Explain the concept of a **graph** mathematically, including the roles of **nodes (vertices)** and **edges**.
  * Introduce GNNs, detailing how their operation differs from traditional neural networks (e.g., CNNs or MLPs) and their key applications.

2. **Explain Graph Operations:**
  * Detail **Graph Convolution**, explaining how it generalizes the concept of convolution from a grid (image) to an irregular graph structure (message passing).
  * Detail **Graph Pooling** (e.g., global or hierarchical pooling) and its importance for summarizing graph-level features while reducing complexity.

3. **PyTorch Geometric Implementation on ZINC:**
  * Utilize the **PyTorch Geometric** library to construct and train a GNN model on the **ZINC dataset**.
  * Design a network architecture suitable for the task of chemical property prediction (a regression task on ZINC).

4. **Performance Evaluation:**
  * Evaluate the GNN model's predictive accuracy on the test set, using appropriate metrics for the regression task (e.g., Mean Absolute Error or Root Mean Squared Error).

5. **Advanced Exploration (Optional):**
  * Implement a **Graph Transformer** model from scratch, referencing the provided research paper.
  * Compare the performance, computational characteristics, and architectural differences of the Graph Transformer model against the standard GNN model.

## Expected Deliverables

1. **Codebase:** Complete, well-documented Python code for the GNN model using PyTorch Geometric and the ZINC dataset. Include the optional implementation of a Graph Transformer model, if completed.
2. Documentation:
  - `README.md`: A summary of the project’s objectives, methodology, and outcomes.
  - `REPORT.md`: A detailed report explaining the theoretical aspects of VAEs, the mathematical foundation of ELBO, and practical applications.

## Resources and References

* **Graph Transformer Paper:** https://proceedings.neurips.cc/paper_files/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf
* **ZINC Dataset Documentation (PyG):** https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ZINC.html
* **PyTorch Geometric GitHub:** https://github.com/pyg-team/pytorch_geometric
* **Review on Graph Neural Networks:** https://arxiv.org/pdf/1812.08434.pdf


# LLM-Based Autonomous Agents with Tool Use and Reflection

**Difficulty**: hard

## Overview

This project focuses on designing and implementing a complete **autonomous agent framework** powered by a Large Language Model (LLM). Unlike simple query-response systems, this agent will be capable of multi-step **planning**, utilizing external **tools** (functions or APIs), and employing **reflection** and **memory** to self-correct and execute complex, sequential tasks in a dynamic environment. The final system should demonstrate robust decision-making across multiple problem domains. 

## Objectives

1. **Research Agentic Architectures:**
  * Thoroughly research leading agentic architectures, such as **ReAct (Reasoning and Acting)**, Chain-of-Thought (CoT) prompting, and self-correction methods like reflection.
  * Explain the role of the action-space and the different types of **memory** (e.g., short-term scratchpad vs. long-term vector database) in the `REPORT.md`.

2. **Implement the Core Agent Loop:**
  * Develop a flexible Python class for the agent that includes the core **Plan $\rightarrow$ Act $\rightarrow$ Reflect** cycle.
  * Ensure the agent can dynamically switch between internal reasoning and external action based on the LLM's prompt analysis.

3. **Integrate External Tools:**
  * Define and integrate at least **three distinct external tools** (e.g., a **Code Interpreter**, a **Web Search API**, and a **Calculator**).
  * The agent must be able to select the correct tool, format the input for that tool, and process the tool's output to continue its reasoning.

4. **Develop Memory and Reflection Mechanisms:**
  * Implement a short-term **memory buffer** (e.g., a list of recent turns) to maintain context.
  * Implement a **reflection module** where the agent critiques the outcome of a failed action or a completed task, generating updated internal plans or knowledge for future use.

5. **Test and Evaluate Robustness:**
  * Define a set of 5-10 complex, multi-step, real-world tasks (e.g., "Find the current stock price of Company X, calculate the 5-year compound annual growth rate, and summarize the CEO's recent public statement").
  * Evaluate the agent's **success rate**, the number of **steps taken** per task, and the effectiveness of the reflection mechanism.

## Expected Deliverables

1. **Codebase:** A complete, well-structured, and object-oriented Python codebase for the autonomous LLM agent framework, including the core loop, tool definitions, and memory components.
2. **Demonstration Logs:** Detailed logs or scripts showcasing the agent successfully completing the defined multi-step evaluation tasks, highlighting the reasoning (ReAct) traces and tool usage.
3. Documentation:
  - `README.md`: A summary of the project’s objectives, methodology, and outcomes.
  - `REPORT.md`: A detailed report explaining the theoretical aspects of VAEs, the mathematical foundation of ELBO, and practical applications.

## Resources and References

* **ReAct: Synergizing Reasoning and Acting in Language Models (Paper):** https://arxiv.org/abs/2210.03629
* **Self-Refine: Iterative Refinement with Self-Feedback (Paper):** https://arxiv.org/abs/2303.17651
* **LLM Agents Overview and Conceptual Tutorial:** https://lilianweng.github.io/posts/2023-06-23-agent/
