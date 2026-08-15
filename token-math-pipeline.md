# Mathematical Walkthrough: LLM Logit Projection, Scaling, and Sampling

This document breaks down the exact mathematical operations performed by a Large Language Model (LLM) during inference when predicting the next token.

**Example Sentence:**
> `"I watched a movie last night. The movie was absolutely ______."`

---

## Table of Contents
1. [Setup & Definitions](#0-setup--definitions)
2. [Step 1: Logit Generation (Matrix Multiplication)](#step-1-logit-generation-matrix-multiplication)
3. [Step 2: Temperature Scaling](#step-2-temperature-scaling)
4. [Step 3: Probability Distribution (Softmax)](#step-3-probability-distribution-softmax)
5. [Step 4: Filtering Candidates (Top-K & Top-P)](#step-4-filtering-candidates-top-k--top-p)
6. [Step 5: Re-Normalization & Final Sampling](#step-5-re-normalization--final-sampling)

---

## 0. Setup & Definitions

* **Context Vector ($\mathbf{h}$):** The output representation from the final Transformer block for the token position `"absolutely"`. Vector dimension $d = 4$.

$$\mathbf{h} = \begin{bmatrix} 1.2 & -0.5 & 2.0 & 0.1 \end{bmatrix}$$

* **Vocabulary ($\mathcal{V}$):** A toy dictionary of size $V = 5$.
  1. `great`
  2. `terrible`
  3. `okay`
  4. `banana`
  5. `the`

* **Output Embedding Matrix ($\mathbf{W}_{\text{out}}$):** Dimensions $d \times V = 4 \times 5$.

$$\mathbf{W}_{\text{out}} = \begin{bmatrix}
2.0 & 1.5 & 0.8 & -1.0 & 0.1 \\
-0.5 & 0.2 & 0.1 & 0.5 & -0.2 \\
1.8 & -1.0 & 0.5 & -2.0 & 0.0 \\
0.4 & 0.1 & -0.2 & 0.3 & 1.2
\end{bmatrix}$$

---

## Step 1: Logit Generation (Matrix Multiplication)

The Language Modeling Head projects the $1 \times d$ context vector $\mathbf{h}$ into vocabulary space via matrix multiplication with $\mathbf{W}_{\text{out}}$ ($d \times V$) to produce raw logits $\mathbf{z}$ ($1 \times V$):

$$\mathbf{z} = \mathbf{h} \cdot \mathbf{W}_{\text{out}}$$

### Logit Calculation per Token:
* **$z_{\text{great}}$**:  
  $$(1.2 \cdot 2.0) + (-0.5 \cdot -0.5) + (2.0 \cdot 1.8) + (0.1 \cdot 0.4) = 2.4 + 0.25 + 3.6 + 0.04 = \mathbf{6.29}$$

* **$z_{\text{terrible}}$**:  
  $$(1.2 \cdot 1.5) + (-0.5 \cdot 0.2) + (2.0 \cdot -1.0) + (0.1 \cdot 0.1) = 1.8 - 0.1 - 2.0 + 0.01 = \mathbf{-0.29}$$

* **$z_{\text{okay}}$**:  
  $$(1.2 \cdot 0.8) + (-0.5 \cdot 0.1) + (2.0 \cdot 0.5) + (0.1 \cdot -0.2) = 0.96 - 0.05 + 1.0 - 0.02 = \mathbf{1.89}$$

* **$z_{\text{banana}}$**:  
  $$(1.2 \cdot -1.0) + (-0.5 \cdot 0.5) + (2.0 \cdot -2.0) + (0.1 \cdot 0.3) = -1.2 - 0.25 - 4.0 + 0.03 = \mathbf{-5.42}$$

* **$z_{\text{the}}$**:  
  $$(1.2 \cdot 0.1) + (-0.5 \cdot -0.2) + (2.0 \cdot 0.0) + (0.1 \cdot 1.2) = 0.12 + 0.1 + 0.0 + 0.12 = \mathbf{0.34}$$

### Raw Logits Vector ($\mathbf{z}$):

$$\mathbf{z} = \begin{bmatrix} 6.29 & -0.29 & 1.89 & -5.42 & 0.34 \end{bmatrix}$$

---

## Step 2: Temperature Scaling

Raw logits are scaled by dividing by the Temperature hyperparameter $T$:

$$\mathbf{z}' = \frac{\mathbf{z}}{T}$$

### Comparison ($T = 0.7$ vs $T = 2.0$):

| Token | Raw Logit ($\mathbf{z}$) | Scaled ($T = 0.7$) | Scaled ($T = 2.0$) |
| :--- | :---: | :---: | :---: |
| `great` | **6.29** | $6.29 / 0.7 = \mathbf{8.986}$ | $6.29 / 2.0 = \mathbf{3.145}$ |
| `okay` | **1.89** | $1.89 / 0.7 = \mathbf{2.700}$ | $1.89 / 2.0 = \mathbf{0.945}$ |
| `the` | **0.34** | $0.34 / 0.7 = \mathbf{0.486}$ | $0.34 / 2.0 = \mathbf{0.170}$ |
| `terrible` | **-0.29** | $-0.29 / 0.7 = \mathbf{-0.414}$ | $-0.29 / 2.0 = \mathbf{-0.145}$ |
| `banana` | **-5.42** | $-5.42 / 0.7 = \mathbf{-7.743}$ | $-5.42 / 2.0 = \mathbf{-2.710}$ |

---

## Step 3: Probability Distribution (Softmax)

The scaled logits $\mathbf{z}'$ are converted into probabilities using the Softmax function:

$$P(y_i) = \frac{e^{z'_i}}{\sum_{j=1}^{V} e^{z'_j}}$$

### Probabilities at $T = 0.7$:
* $e^{8.986} \approx 7990.47$
* $e^{2.700} \approx 14.88$
* $e^{0.486} \approx 1.63$
* $e^{-0.414} \approx 0.66$
* $e^{-7.743} \approx 0.0004$

Sum of exponentials: $\sum e^{z'} = 8007.64$

* **$P(\text{great})$**: $7990.47 / 8007.64 = \mathbf{99.79\%}$
* **$P(\text{okay})$**: $14.88 / 8007.64 = \mathbf{0.19\%}$
* **$P(\text{the})$**: $1.63 / 8007.64 = \mathbf{0.02\%}$
* **$P(\text{terrible})$**: $\approx \mathbf{0.008\%}$
* **$P(\text{banana})$**: $\approx \mathbf{0.000005\%}$

*(Lower temperature sharpens the distribution, giving the top token total dominance).*

---

## Step 4: Filtering Candidates (Top-K & Top-P)

Assume a moderate temperature configuration where candidate probabilities are more balanced:

$$\mathbf{P} = \{ \text{great}: 0.70,\, \text{okay}: 0.18,\, \text{terrible}: 0.08,\, \text{the}: 0.035,\, \text{banana}: 0.005 \}$$

### Case A: Top-K Filtering ($K = 2$)
Keep only the top $K = 2$ candidates by rank and drop the rest:

$$\mathcal{V}_{\text{filtered}} = \{ \text{great}, \text{okay} \}$$

### Case B: Top-P / Nucleus Filtering ($P = 0.90$)
Sort tokens in descending order and compute cumulative probability until the sum reaches $0.90$:

1. $P(\text{great}) = 0.70$  
   *(Cumulative sum = 0.70, which is less than 0.90)* $\rightarrow$ **Keep**
2. $P(\text{okay}) = 0.18$  
   *(Cumulative sum = 0.70 + 0.18 = 0.88, which is less than 0.90)* $\rightarrow$ **Keep**
3. $P(\text{terrible}) = 0.08$  
   *(Cumulative sum = 0.88 + 0.08 = 0.96, which reaches 0.90)* $\rightarrow$ **Keep and Stop**

$$\mathcal{V}_{\text{filtered}} = \{ \text{great}, \text{okay}, \text{terrible} \}$$

---

## Step 5: Re-Normalization & Final Sampling

Taking the Top-P filtered set $\{ \text{great}: 0.70,\, \text{okay}: 0.18,\, \text{terrible}: 0.08 \}$:

### 1. Sum of Filtered Probabilities:
$$S = 0.70 + 0.18 + 0.08 = 0.96$$

### 2. Re-Normalize (Scale to sum to 100%):
* $P_{\text{new}}(\text{great}) = 0.70 / 0.96 = \mathbf{72.92\%}$
* $P_{\text{new}}(\text{okay}) = 0.18 / 0.96 = \mathbf{18.75\%}$
* $P_{\text{new}}(\text{terrible}) = 0.08 / 0.96 = \mathbf{8.33\%}$

### 3. Cumulative Interval Distribution for Sampling:
* $[0.0000, 0.7292) \longrightarrow \text{great}$
* $[0.7292, 0.9167) \longrightarrow \text{okay} \quad (0.7292 + 0.1875 = 0.9167)$
* $[0.9167, 1.0000) \longrightarrow \text{terrible}$

### 4. Sampling Selection:
Draw a random number $r \sim U(0, 1)$.  
If $r = 0.82$, it falls within the interval $[0.7292, 0.9167)$.

**Selected Next Token:** **`okay`**
