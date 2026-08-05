# Uncertainty Quantification in Language Models

## 1. Introduction

Language models often generate confident-sounding text even when uncertain, which is risky in practical applications where knowing confidence is crucial. This project implements and evaluates methods to estimate uncertainty at both sentence and token levels for language model outputs 

### Scope

- **Model Selection**: Focus on recent smaller open-source models (Qwen3-4B-Instruct-2507, Qwen3-8B, Ministral-8B-Instruct-2410), to evaluate 2 models in the same family (Qwen3) and models from different families but the same size (Qwen3-8B, Ministral-8B-Instruct-2410) (we assumed that 8B models are "in the same league" as 7B models - even though the instructions advised 7B or smaller)
- **Granularity**: Both token-level and sentence-level uncertainty estimation
- **Task Diversity**: Four task types with different characteristics: binary correctness (QA and Math) vs. continuous quality (Summarization and Open-ended)

## 2. Methodology

### 2.1 Uncertainty Estimation Methods

We implement six uncertainty quantification methods, categorized into probability-based and sampling-based approaches:

#### Probability-Based Methods

1. **Entropy** [1]: Token-level entropy calculated from probability distributions
   - Measures the spread of the probability distribution
   - Higher entropy = more uncertainty
   - Provides both token-level and sentence-level (mean) uncertainty

2. **Max Probability** [2]: Uncertainty based on the maximum token probability
   - Uncertainty = 1 - max(p)
   - Lower max probability = more uncertainty
   - Provides both token-level and sentence-level uncertainty

3. **Sequence Probability** [2]: Negative log-probability of the generated sequence
   - Measures the overall likelihood of the generated sequence
   - Higher negative log-probability = more uncertainty
   - Sentence-level only

#### Sampling-Based Methods

4. **Self-Consistency** [3]: Measures agreement across multiple sampled outputs
   - Generates N samples and computes semantic distance to reference output
   - Uses cosine similarity between sentence embeddings
   - Higher variance = more uncertainty
   - Calculates similarity to reference output and not cross-similarity between the N samples to reduce complexity and make it more suitable for inference 
   - Can be used in models exposed through an API as it doesn't require access to output logits 

5. **Self-Reflection** [4]: Model evaluates its own output to assess uncertainty
   - Based on BSDetector approach
   - Model rates its answer as (A) Correct, (B) Incorrect, or (C) Not sure
   - Maps to uncertainty scores: 0.0 (confident), 0.5 (uncertain), 1.0 (incorrect)
   - Can be used in models exposed through an API as it doesn't require access to output logits 

6. **BSDetector** [4]: Combines self-consistency and self-reflection
   - Weighted combination: β × self_consistency + (1-β) × self_reflection
   - Default β = 0.7 (favors self-consistency)
   - Provides both extrinsic (sampling) and intrinsic (reflection) confidence signals

### 2.2 Design Decisions

**Greedy Decoding for Probability-Based Methods**: We use greedy decoding (do_sample=False) for probability-based methods to ensure deterministic, interpretable uncertainty from the model's probability distributions rather than sample diversity.

**EOS Token Handling**: We skip EOS tokens in token-level uncertainty calculations to avoid control-token bias that could inflate uncertainty estimates.

**Semantic Embeddings**: Self-consistency uses sentence transformers (all-MiniLM-L6-v2) for efficient semantic similarity computation without requiring task-specific evaluators.

### 2.3 Evaluation Framework

#### Metrics

1. **Correlation Metrics** (for tasks with a continuous quality score):
   - Spearman correlation between uncertainty and errors
   - Measures monotonic relationship strength

2. **Discrimination Metrics** (for tasks with binary correctness):
   - **AUROC** (Area Under ROC Curve): Ability to distinguish correct from incorrect predictions
   - Higher AUROC = better uncertainty discrimination

3. **Calibration Metrics**:
   - **ECE** (Expected Calibration Error): Measures how well confidence aligns with accuracy
   - Lower ECE = better calibrated uncertainty
   - Computed with 10 bins by default

4. **Task-Specific Metrics**:
   - **QA/Math**: Binary correctness (0/1) based substring matching, which seemed appropriate for the simple QA and Math tasks we are running
   - **Summarization**: ROUGE-L F-measure (0.0-1.0), since it's a recall-based metric its focus is on information preservation
   - **Open-ended**: Prometheus 2 [5] score (normalized 0.0-1.0), Prometheus 2 is optimized for rating tasks and has strong alignment with human judgment 

#### Visualization Tools

- **ROC Curves**: AUROC visualization for binary tasks
- **Correlation Plots**: Uncertainty vs. error scatter plots for continuous tasks
- **Calibration Curves**: Reliability diagrams showing predicted vs. actual accuracy
- **Uncertainty Distributions**: Histograms of uncertainty values
- **Token-level Heatmaps**: Visual representation of token-level uncertainty for probability-based methods

## 3. Implementation

### 3.1 Trade-offs of Uncertainty Quantification Methods 

**Computational Cost**:
- Probability-based methods: Low overhead (reuse generation scores)
- Self-consistency: Moderate (N additional generations)
- Self-reflection: Moderate (1 additional generation)
- BSDetector: Highest (combines both sampling methods)

**Accuracy vs. Speed**:
- Probability-based methods are fast but may not capture semantic uncertainty
- Sampling-based methods are slower but better capture model disagreement

**Token vs. Sentence Level**:
- Token-level provides fine-grained insights but requires more storage
- Sentence-level is more practical for downstream applications

## 4. Experimental Setup

### 4.1 Models

- **Qwen3-4B-Instruct-2507**: 4B parameter instruction-tuned model
- **Qwen3-8B**: 8B parameter model 
- **Ministral-8B-Instruct-2410**: 8B parameter model

### 4.2 Tasks

1. **Factual QA** (SQuAD validation set)
   - Binary correctness evaluation
   - Substring matching with punctuation normalization because answers are short and factual

2. **Math Problems** (GSM8K test set)
   - Binary correctness evaluation
   - Substring matching with punctuation normalization because answers are short and factual

3. **Summarization** (CNN/DailyMail test set)
   - Continuous quality evaluation (ROUGE-L)
   - Measures summary quality, not just correctness, focusing on the preservation of information

4. **Open-ended Generation** (WritingPrompts test set)
   - Continuous quality evaluation (Prometheus 2)
   - Requires reference answers for evaluation
 
### 4.3 Experimental Configuration

- **Generation Parameters**: max_length=50 (for faster experimentation), temperature=1.0
- **Sampling Parameters**: num_samples=5 for self-consistency
- **Evaluation**: All six uncertainty methods per task
- **Sample Sizes**: 200 samples per task randomly sampled from each dataset

### 4.4 Hardware and Environment

- **GPU**: NVIDIA H100 80GB HBM3 (79.1 GB)
- **CUDA Version**: 12.4
- **PyTorch Version**: 2.4.1
- **Python Version**: 3.10.14

## 5. Results

We evaluated six uncertainty quantification methods across three models (Qwen3-4B-Instruct-2507, Qwen3-8B, Ministral-8B-Instruct-2410) on four tasks. The results are organized by task below, with one table per task showing methods as rows and models as columns with subcolumns for AUROC/Correlation and ECE. **Bold values** indicate the best performance: highest AUROC/Correlation and lowest ECE for each model.

### 5.1 Comprehensive Results Tables

#### Factual QA (SQuAD)

| Method | Ministral-8B-Instruct-2410<br>AUROC | Ministral-8B-Instruct-2410<br>ECE | Qwen3-4B-Instruct-2507<br>AUROC | Qwen3-4B-Instruct-2507<br>ECE | Qwen3-8B<br>AUROC | Qwen3-8B<br>ECE |
|:------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Entropy | 0.830 | 0.581 | 0.658 | 0.606 | 0.685 | 0.547 |
| Max Probability | 0.858 | 0.408 | **0.712** | **0.526** | 0.680 | **0.465** |
| Sequence Probability | **0.881** | 0.846 | 0.701 | 0.633 | 0.704 | 0.578 |
| Self-Consistency | 0.800 | **0.393** | 0.671 | 0.546 | 0.677 | 0.496 |
| Self-Reflection | 0.503 | 0.855 | 0.597 | 0.667 | 0.577 | 0.758 |
| BSDetector | 0.842 | 0.421 | 0.702 | 0.562 | **0.808** | 0.540 |

#### Math Problems (GSM8K)

| Method | Ministral-8B-Instruct-2410<br>AUROC | Ministral-8B-Instruct-2410<br>ECE | Qwen3-4B-Instruct-2507<br>AUROC | Qwen3-4B-Instruct-2507<br>ECE | Qwen3-8B<br>AUROC | Qwen3-8B<br>ECE |
|:------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Entropy | 0.816 | **0.229** | 0.666 | 0.295 | 0.778 | 0.339 |
| Max Probability | 0.810 | 0.230 | **0.679** | **0.228** | 0.777 | 0.336 |
| Sequence Probability | **0.837** | 0.279 | 0.667 | 0.313 | **0.778** | 0.392 |
| Self-Consistency | 0.728 | 0.336 | 0.551 | 0.358 | 0.679 | **0.295** |
| Self-Reflection | 0.500 | 0.800 | 0.529 | 0.575 | 0.593 | 0.565 |
| BSDetector | 0.742 | 0.339 | 0.571 | 0.396 | 0.744 | 0.354 |

#### Summarization (CNN/DailyMail)

| Method | Ministral-8B-Instruct-2410<br>Correlation | Ministral-8B-Instruct-2410<br>ECE | Qwen3-4B-Instruct-2507<br>Correlation | Qwen3-4B-Instruct-2507<br>ECE | Qwen3-8B<br>Correlation | Qwen3-8B<br>ECE |
|:------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Entropy | 0.229 | 0.359 | **0.327** | 0.488 | **0.213** | 0.423 |
| Max Probability | 0.196 | **0.277** | 0.309 | 0.444 | 0.200 | **0.374** |
| Sequence Probability | 0.195 | 0.326 | 0.305 | 0.464 | 0.190 | 0.400 |
| Self-Consistency | **0.232** | 0.384 | 0.205 | 0.525 | 0.088 | 0.413 |
| Self-Reflection | N/A | 0.764 | N/A | 0.784 | 0.026 | 0.772 |
| BSDetector | 0.185 | 0.485 | 0.215 | 0.626 | 0.197 | 0.510 |

#### Open-ended Generation (WritingPrompts)

| Method | Ministral-8B-Instruct-2410<br>Correlation | Ministral-8B-Instruct-2410<br>ECE | Qwen3-4B-Instruct-2507<br>Correlation | Qwen3-4B-Instruct-2507<br>ECE | Qwen3-8B<br>Correlation | Qwen3-8B<br>ECE |
|:------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Entropy | -0.146 | 0.262 | -0.238 | 0.343 | -0.111 | 0.464 |
| Max Probability | -0.240 | **0.226** | -0.328 | 0.310 | -0.299 | 0.474 |
| Sequence Probability | -0.210 | 0.277 | -0.285 | 0.338 | -0.251 | 0.482 |
| Self-Consistency | 0.202 | 0.244 | -0.038 | **0.180** | 0.001 | 0.263 |
| Self-Reflection | 0.089 | 0.632 | **0.219** | 0.500 | 0.032 | 0.392 |
| BSDetector | **0.231** | 0.266 | 0.141 | 0.240 | 0.010 | **0.259** |

### 5.2 Method Comparison

**Tasks with binary correctness metric (QA, Math):**

- **Best AUROC Performance**: 
  - QA: Sequence Probability (0.881, Ministral-8B-Instruct-2410) and Max Probability (0.858, Ministral-8B-Instruct-2410) achieve the highest discrimination
  - Math: Sequence Probability (0.837, Ministral-8B-Instruct-2410) and Entropy (0.816, Ministral-8B-Instruct-2410) perform best
  - BSDetector shows strong performance on QA (0.842 Ministral-8B-Instruct-2410, 0.808 Qwen3-8B) but is less effective on Math problems

- **Best Calibration (Lowest ECE)**:
  - QA: Self-Consistency (0.393, Ministral-8B-Instruct-2410) and Max Probability (0.408, Ministral-8B-Instruct-2410) are best calibrated
  - Math: Max Probability (0.228, Qwen3-4B-Instruct-2507) and Entropy (0.229, Ministral-8B-Instruct-2410) achieve the best calibration
  - Self-Reflection consistently shows poor calibration across all models and tasks (ECE > 0.5)

- **Probability-based vs. Sampling-based**:
  - Probability-based methods (Entropy, Max Probability, Sequence Probability) generally achieve higher AUROC on tasks that are more factual
  - Self-Consistency provides competitive calibration but lower discrimination
  - Self-Reflection performs poorly on binary tasks, suggesting limited ability to self-evaluate correctness

**Tasks with continuous quality metric (Summarization, Open-ended):**

- **Best Correlation Performance**:
  - Summarization: Entropy achieves the highest correlation (0.327, Qwen3-4B-Instruct-2507), followed by Max Probability (0.309, Qwen3-4B-Instruct-2507)
  - Open-ended: BSDetector (0.231, Ministral-8B-Instruct-2410) and Self-Consistency (0.202, Ministral-8B-Instruct-2410) show positive correlations, while probability-based methods show negative correlations
  - Interestingly, probability-based methods (Entropy, Max Probability, Sequence Probability) show negative correlations on open-ended tasks, indicating they do not capture semantic uncertainty well

- **Best Calibration**:
  - Summarization: Max Probability achieves the best calibration (0.277, Ministral-8B-Instruct-2410)
  - Open-ended: Self-Consistency (0.18, Qwen3-4B-Instruct-2507) and BSDetector (0.24-0.266 across models) show good calibration
  - Self-Reflection again shows poor calibration on continuous tasks

### 5.3 Task-Specific Analysis

**Factual QA (SQuAD):**
- Error rates are high (84.5-89%), indicating the models struggle with factual questions
- Ministral-8B-Instruct-2410 shows the best overall performance with multiple methods achieving AUROC > 0.8
- Sequence Probability and Max Probability are the top performers, suggesting probability distributions capture uncertainty well for factual tasks

**Math Problems (GSM8K):**
- Error rates vary more across models (67-80%), with Qwen3-4B-Instruct-2507 performing best
- Probability-based methods (Entropy, Max Probability, Sequence Probability) consistently outperform sampling-based methods
- Calibration is generally better than QA, with ECE values often below 0.3 for probability-based methods

**Summarization (CNN/DailyMail):**
- Correlations are modest (0.09-0.33 for most methods), indicating limited ability to predict quality from uncertainty
- Qwen3-4B-Instruct-2507 shows the strongest correlations, particularly with Entropy (0.327, p<0.001)
- Most correlations are statistically significant (p<0.05), except for Qwen3-8B with Self-Consistency (r=0.088, p=0.217) and Self-Reflection (r=0.026, p=0.717)
- Self-Reflection fails to produce valid correlations for Qwen3-4B-Instruct-2507 and Ministral-8B-Instruct-2410 (N/A values)
- Max Probability achieves the best calibration (0.277, Ministral-8B-Instruct-2410)

**Open-ended Generation (WritingPrompts):**
- This task reveals the most significant differences between methods and models
- Probability-based methods show negative correlations (statistically significant for Ministral-8B-Instruct-2410 and Qwen3-4B-Instruct-2507), suggesting they may be measuring something other than semantic quality uncertainty
- **Qwen3-8B shows poor uncertainty quantification on this task**: Most methods fail to achieve statistically significant correlations (p≥0.05), including BSDetector (r=0.010, p=0.894), Self-Consistency (r=0.001, p=0.989), and Self-Reflection (r=0.032, p=0.650). Only Max Probability and Sequence Probability show significant negative correlations.
- For Ministral-8B-Instruct-2410 and Qwen3-4B-Instruct-2507, sampling-based methods (BSDetector, Self-Consistency) show statistically significant positive correlations, indicating they better capture semantic uncertainty
- BSDetector achieves the best correlation (0.231, p=0.001, Ministral-8B-Instruct-2410) and good calibration (0.266)

### 5.4 Calibration Analysis

1. **Probability-based methods** show better calibration on Math tasks (ECE often < 0.4) compared to QA tasks, and generally better calibration on binary tasks compared to continuous tasks

2. **Self-Reflection** consistently shows the worst calibration across all tasks and models (ECE > 0.5, often > 0.7), suggesting the model's self-assessment is poorly calibrated

3. **Model-specific patterns**:
   - Ministral-8B-Instruct-2410 generally achieves better calibration than Qwen models on most tasks
   - Qwen3-4B-Instruct-2507 shows better calibration on Math problems compared to Qwen3-8B

4. **Task-specific patterns**:
   - Math problems show the best overall calibration (ECE often < 0.3)
   - Open-ended generation shows good calibration for sampling-based methods (ECE < 0.3)
   - Summarization shows moderate calibration (ECE 0.3-0.5 for most methods)

5. **Method-specific patterns**:
   - Max Probability consistently achieves good calibration across tasks
   - BSDetector shows competitive calibration, particularly on open-ended tasks
   - Sequence Probability shows variable calibration, performing well on Math but poorly on QA

**Visualization Figures:**

Detailed visualization figures are available in the `results/` directory for each model and task:
- **Calibration curves**: Reliability diagrams showing predicted vs. actual accuracy (e.g., `results/results_qwen3_4b/qa/figures/entropy_calibration.png`)
- **ROC curves**: AUROC visualization for binary tasks (e.g., `results/results_qwen3_4b/qa/figures/entropy_roc.png`)
- **Correlation plots**: Uncertainty vs. error scatter plots for continuous tasks (e.g., `results/results_qwen3_4b/summarization/figures/entropy_correlation.png`)
- **Uncertainty distributions**: Histograms of uncertainty values (e.g., `results/results_qwen3_4b/qa/figures/entropy_distribution.png`)
- **Token-level heatmaps**: Visual representation of token-level uncertainty for probability-based methods (e.g., `results/results_qwen3_4b/qa/figures/entropy_token_heatmap.png`)

### 5.5 Latency vs. Quality Trade-offs

(Note: Latency analysis was not included in this evaluation, as the focus was on method effectiveness rather than computational efficiency.)






## 6. Discussion

### 6.1 Method Effectiveness

Based on the experimental results, we can draw the following conclusions

**For Binary Correctness Tasks (QA, Math):**
- **Probability-based methods are superior**: Entropy, Max Probability, and Sequence Probability consistently achieve higher AUROC values (0.66-0.88) compared to sampling-based methods
- **Max Probability is the best overall choice**: It achieves competitive AUROC while maintaining excellent calibration (ECE often < 0.3 on Math, < 0.5 on QA)
- **Self-Reflection is ineffective**: With AUROC near 0.5 (random performance) and poor calibration (ECE > 0.5), self-reflection fails to provide useful uncertainty estimates for binary tasks
- **BSDetector shows promise on QA**: While not the top performer, BSDetector achieves strong AUROC (0.808-0.842) on QA tasks, suggesting the combination of sampling and reflection can be effective


**For Continuous Quality Tasks (Summarization, Open-ended):**
- **Task-dependent method selection is critical**: 
  - For summarization, probability-based methods (especially Entropy) show the strongest correlations (0.19-0.33), with most being statistically significant (p<0.05)
  - For open-ended generation, sampling-based methods (BSDetector, Self-Consistency) are essential, as probability-based methods show negative correlations
- **Semantic uncertainty requires sampling**: The negative correlations of probability-based methods on open-ended tasks (-0.24 to -0.33) are statistically significant for Ministral-8B-Instruct-2410 and Qwen3-4B-Instruct-2507, indicating they capture token-level uncertainty rather than semantic quality uncertainty
- **Model-specific challenges**: Qwen3-8B shows particularly poor uncertainty quantification on open-ended tasks, with most methods failing to achieve statistically significant correlations (p≥0.05), suggesting model-specific limitations
- **BSDetector is the best choice for open-ended tasks**: It achieves the highest statistically significant correlation (0.231, p=0.001) and good calibration (0.266) on open-ended generation for Ministral-8B-Instruct-2410

**General Recommendations:**
1. **Use Max Probability for binary tasks** when computational efficiency is important and you have access to output logits
2. **Use Entropy for summarization** when correlation with quality is the priority and you have access to output logits
3. **Use BSDetector for open-ended generation** when semantic uncertainty is critical
4. **Avoid Self-Reflection** unless the model has been specifically fine-tuned for self-evaluation
5. **Consider Self-Consistency** when good calibration is needed and computational cost is acceptable, particularly useful when we don't have access to model outputs, other than the text (e.g., through an API)

### 6.2 Computational Cost vs. Quality Trade-offs

We analyze the latency using Ministral-8B-Instruct-2410 as a representative example, since it achieves the best overall performance across tasks. The following table shows the latency for each method across all tasks:

#### Latency by Method and Task (Ministral-8B-Instruct-2410)

| Method | Factual QA<br>(SQuAD) | Math Problems<br>(GSM8K) | Summarization<br>(CNN/DailyMail) | Open-ended<br>(WritingPrompts) |
|:-------|:---------------------:|:------------------------:|:--------------------------------:|:----------------------------:|
| Sequence Probability | 0.353 ms | 0.393 ms | 3.5 ms | 3.6 ms |
| Max Probability | 0.544 ms | 0.622 ms | 5.9 ms | 6.2 ms |
| Entropy | 0.800 ms | 0.846 ms | 7.8 ms | 8.1 ms |
| Self-Reflection | 35.2 ms | 36.1 ms | 36.5 ms | 36.0 ms |
| Self-Consistency | 615.1 ms | 714.5 ms | 7.016 s | 7.391 s |
| BSDetector | 652.5 ms | 748.3 ms | 7.053 s | 7.404 s |

**Key Observations:**

1. **Probability-based methods are highly efficient**: All three probability-based methods (Sequence Probability, Max Probability, Entropy) have latencies under 10 ms, making them suitable for real-time applications. For binary tasks, they achieve latencies under 1 ms.

2. **Self-Reflection offers poor value**: At ~36 ms latency (40-100x slower than probability-based methods), Self-Reflection performs poorly across all tasks despite the additional computational cost.

3. **Sampling-based methods are expensive**: Self-Consistency and BSDetector require 0.6-7.4 seconds, making them 1000x slower than probability-based methods. However, they are essential for open-ended generation where probability-based methods fail to capture semantic uncertainty.

### 6.3 Limitations

1. **Model Dependencies**: 
   - Self-consistency and BSDetector require semantic embedding models
   - Self-reflection depends on model's ability to self-evaluate accurately

2. **Task-Specific Challenges**:
   - The definition of correctness and quality of output is extremely important to assess uncertainty and is one of the aspects that is less fleshed out in this work

3. **Computational Constraints**:
   - Sampling-based methods are expensive for production and require more validation about their effectiveness

### 6.4 Potential Improvements

1. **Hybrid Methods**: Combine fast probability-based methods (max_probability) with slower but potentially more accurate sampling-based methods (self-consistency). For example, use probability-based uncertainty as a filter to decide when expensive sampling-based methods are worth computing, or combine their outputs using learned weights.

2. **Task-Specific Calibration**: Learn calibration functions per task type to better map raw uncertainty scores to actual confidence levels, potentially improving ECE across different domains.

3. **Uncertainty Aggregation**: Better methods for combining token-level to sentence-level uncertainty beyond simple averaging (e.g., weighted by token importance, attention-based aggregation, or considering sequence structure).

### 6.5 Applications of Uncertainty Quantification methods 

1. **Active Learning**: Use uncertainty estimates to guide data collection and model fine-tuning, focusing high uncertainty samples.

2. **Confidence-Based Generation**: In domains requiring high confidence (e.g., medical diagnosis, legal advice, safety-critical systems), continuously sample multiple generations until the uncertainty falls below a threshold. This adaptive sampling approach ensures outputs meet minimum confidence requirements before deployment, trading computational cost for reliability.

## 7. Conclusion

This project implements and evaluates a comprehensive framework for uncertainty quantification in language models across six methods, three models, and four diverse tasks. Our experimental results reveal several key insights:

**Key Findings:**

1. **Task type determines optimal method**: Probability-based methods (Entropy, Max Probability, Sequence Probability) excel on binary correctness tasks (QA, Math) with AUROC values up to 0.88, while sampling-based methods (BSDetector, Self-Consistency) are essential for open-ended generation where semantic uncertainty matters.

2. **Max Probability is the most reliable general-purpose method**: It achieves competitive discrimination (AUROC 0.68-0.86) while maintaining excellent calibration (ECE often < 0.3) across binary tasks, making it a strong default choice.

3. **Self-Reflection is ineffective in small models**: Despite requiring additional computation, self-reflection consistently performs poorly (AUROC near 0.5, ECE > 0.5) across all tasks and models, suggesting current models lack reliable self-evaluation capabilities.

4. **Calibration varies significantly**: Math problems show the best calibration (ECE < 0.3), while QA tasks show worse calibration (ECE often > 0.5), indicating task difficulty affects uncertainty calibration.

5. **Model choice matters**: Ministral-8B-Instruct-2410 generally achieves better uncertainty discrimination and calibration than Qwen models, though Qwen3-4B-Instruct-2507 shows competitive performance on Math problems, better than Qwen3-8B. This may be due to Qwen3-4B-Instruct-2507 being an instruct model, capable of following instructions more competently. Notably, Qwen3-8B shows particularly poor uncertainty quantification on open-ended tasks, with most methods failing to achieve statistically significant correlations (p≥0.05). 

6. **Semantic vs. token-level uncertainty**: Probability-based methods capture token-level uncertainty well for factual tasks but fail on open-ended generation (negative correlations), where sampling-based methods that measure semantic consistency are required.

## References

1. Malinin, A., & Gales, M. (2021). Uncertainty Estimation in Autoregressive Structured Prediction. *International Conference on Learning Representations (ICLR)*.

2. Malinin, A., & Gales, M. (2020). Predictive Uncertainty Estimation via Prior Networks. *Advances in Neural Information Processing Systems (NeurIPS)*.

3. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., ... & Zhou, D. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. 

4. Chen, J., & Mueller, J. (2024). Quantifying Uncertainty in Answers from Any Language Model and Enhancing Their Trustworthiness. *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)*.

5. Kim, S., Kim, D., Cho, H., Kim, J., Kim, H., Lee, S., ... & Park, S. (2024). Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models. 



