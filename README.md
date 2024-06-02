# Can LLMs think Creatively?


This project analyses the creative reasoning capabilities of Large Language Models (LLMs). We evaluate instruction based models, FlanT5 and RoBERTa on creative puzzles that require deviation from conventional logical thinking. We refer the Brainteaser paper for the baselines. To enhance the performance of these models, we make 2 contributions-
1. Fine-tuning techniques, including Full fine-tuning and Parameter Efficient Fine Tuning (PEFT)
2. Making adversarial versions of the dataset to increase variability of fine-tuning data

Motivation: [BRAINTEASER](https://semevalbrainteaser.github.io/) SemEval 2024 task


**Data:** 
1. Evaluation dataset: BRAINTEASER dataset which consists of multiple-choice sentence puzzles (120 QA-pairs)  
Example puzzle: What animal has no wings, but yet will fly?   
Answer: A caterpillar    
Common Mistake: An eagle.

2. Fine-tuning dataset: We took Puzzles from RIDDLESENSE dataset. Baseline models struggled to consistently apply reasoning across puzzle variants. To alleviate this, we create adversarial versions of the RIDDLESENSE dataset.

**Experiments:**
1. Fine-tuning: For smaller models such as RoBERTa-L (354M) and FlanT5-L (780M). Our FlanT5 base-lines follow a prompting method, which involves presenting both the question and answer options jointly to the model and generating the symbol associated with the chosen answer.
We fine-tune RoBERTa-L for 3 epochs with a learning rate of 1e −6 and batch size of 4. 
We fine-tune FlanT5 models for 3 epochs, with a learning rate of 3e −4, weight decay of 0.01 and batch size of 8.

2. Parameter- Efficient Fine-Tuning (PEFT): Since full fine-tuning of larger models (>1B parameters) is not practical, we instead used Parameter-Efficient Fine-Tuning (PEFT) techniques. We used Low-Rank Adaptation (LoRA) for fine-tuning the FlanT5-XL model (3B), updating only the rank-decomposition matrices within each Transformer layer while keeping the original model parameters static.  
For fine-tuning using LoRA, we set rank r = 16, scaling factor α = 32, and dropout rate of 0.05. We apply LoRA to just the query and value matrices of self-attention heads. 

3. Adversarial Versions of RIDDLESENSE:  We leveraged GPT-4 Turbo to rephrase questions for creating adversarial versions of dataset.  
We created two variants of each puzzle- Semantic Reconstruction, which rephrased questions while retaining the same answers, and Context Reconstruction, which altered the scenarios or settings of the puzzles without changing underlying logic. 

4. Process of Elimination (PoE): Humans typically answer multiple-choice question answers (MCQA) by first eliminating incorrect options and then choosing from the remaining ones. To mimic this process in language models, we implement the two-step Process of Elimination (POE) method as an alternative method MCQA tasks.  
The first step of the POE method comprises scoring each option to identify and eliminate incorrect choices, followed by a second step where the model predicts the final answer from the remaining options by masking incorrect options from the first step.



**Evaluation:**
1. Instance-based accuracy - Measure performance on individual puzzles
2. Group-based accuracy - measure consistency across all variations, original, semantic and syntactic reconstructions. 
   - Original and Semantic 
   - Original Semantic and Context

These metrics provide a comprehensive assessment of the models' creative reasoning abilities and robustness under different testing conditions. All assessments were conducted in zero-shot settings.
All evaluations are conducted in zero-shot setting.

**Results:**
1. Baseline - Baseline implementations 
- FlanT5 (Large and XL variants), an instruction-fine-tuned version of T5
- RoBERTa-L, the vanilla RoBERTa model
- RoBERTa-L (CSKG), RoBERTa-L fine-tuned on the synthetic QA pairs generated from the CommonSense Knowledge Graph (CSKG)  

   Surprisingly, baseline FlanT5 models, including FlanT5-XL with 3B parameters, performed worse than vanilla RoBERTa-L. FlanT5 models underperform significantly despite being larger models  
   RoBERTa-L (CSKG) underperformed compared to its vanilla counterpart. This validates the hypothesis that commonsense knowledge may not always align with the requirements of lateral thinking puzzles  

| Model                              | OG        | SR        | CR        | OG & SR   | OG & SR & CR | Overall   |
|------------------------------------|-----------|-----------|-----------|-----------|--------------|-----------|
| RoBERTa-L                          | 44.97     | 41.42     | 47.92     | 33.13     | 20.71        | 44.77     |
| RoBERTa-L (CSKG)                   | 34.91     | 36.68     | 46.15     | 31.36     | 17.15        | 39.25     |
| RoBERTa-L (RS)                     | 42.11     | 45.93     | 54.55     | 37.32     | 27.75        | **47.53**     |
| FlanT5-L                           | 17.16     | 17.75     | 23.08     | 11.24     | 3.55         | 19.33     |
| FlanT5-L (RS)                      | 51.48     | 55.03     | 53.85     | 44.97     | 30.77        | 53.45     |
| FlanT5-L (RS<sub>adv</sub>)        | 57.4      | 59.76     | 56.21     | 50.3      | 33.73        | **57.79** |
| FlanT5-XL                          | 25.44     | 23.66     | 35.5      | 17.15     | 13.61        | 28.2      |
| FlanT5-XL (RS)                     | 60.95     | 58.57     | 56.8      | 52.66     | 38.46        | 58.78     |
| **FlanT5-XL (RS<sub>adv</sub>)**   | **65.09** | **62.72** | **67.46** | **55.62** | **45.56**    | **65.09** |
| FlanT5-XL (RS<sub>; LoRA)          | 52.07     | 49.7      | 60.95     | 44.38     | 34.32        | 58.24     |
| FlanT5-XL (RS<sub>adv</sub>; LoRA) | 47.34     | 50.3      | 55.03     | 40.83     | 28.99        | 50.89     |
| ChatGPT*                           | 60.77     | 59.33     | 67.94     | 50.72     | 39.71        | 62.68     |

*ChatGPT results are taken from original [BrainTeaser repo](https://github.com/1171-jpg/BrainTeaser)
2. Impact of PoE: Mask Accuracy is the ratio of instances that retain their correct option after elimination in the first step. While our fine-tuned models have > 90% masking accuracies, we still do not observe a significant improvement in performance for any models. This is consistent with the finding from Ma and Du (2023) that POE performs better on tasks that require logical reasoning, where elimination is straightforward based on logic or factual information.



**Conclusion:**
Based on our analysis, we fine-tune our models on the RIDDLESENSE dataset and our adversarial version of the RIDDLESENSE dataset, leading to significant improvements in the performance of FlanT5 models. Notably, FlanT5-XL RSadv with an over-all accuracy of 65.09% outperforms ChatGPT with an overall accuracy of 62.68%. 


**Notes:**
- The training and evaluation data for the ***Sentence Puzzle*** task is in `data`, converted from `.npy` (original data format) to `.json` using `load_data.py`.
- `flant5-inference.py` was used to produce FLanT5 results
- `roberta-inference.py` was used to produce Roberta results
- `roberta-cskg-baseline.py` was used to reproduce RoBERTa-L (CSKG) results. RoBERTa baselines are based on the code for the paper [Knowledge-driven Data Construction for Zero-shot Evaluation in Commonsense Question Answering](https://github.com/Mayer123/HyKAS-CSKG/tree/main). Weights for RoBERTa (CSKG) were also obtained from the same repository.
- `Metrics_Calculation.ipynb` was used to calculate metrics.
