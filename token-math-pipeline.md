# Mathematical Walkthrough: How an LLM Chooses the Next Token

This document breaks down the mathematical steps an LLM follows during inference when predicting the next token.

> **Example sentence:**  
> "I watched a movie last night. The movie was absolutely ______."

This is a toy example. Real LLMs use much larger vectors and vocabularies.

---

## Table of Contents

1. Setup & Definitions
2. Step 1: Logit Generation
3. Step 2: Temperature Scaling
4. Step 3: Probability Distribution with Softmax
5. Step 4: Filtering with Top-K and Top-P
6. Step 5: Re-Normalization and Sampling
7. The Complete Pipeline

---

## 0. Setup & Definitions

### Context Vector

The final Transformer block produces a hidden representation for the current token position, `"absolutely"`.

For this toy example, the vector has 4 dimensions:

```text
h = [1.2, -0.5, 2.0, 0.1]
```

### Vocabulary

We use a toy vocabulary containing five possible next tokens:

1. `great`
2. `terrible`
3. `okay`
4. `banana`
5. `the`

So the vocabulary size is **5**.

### Output Projection / Unembedding Matrix

The Language Modeling Head projects the context vector into vocabulary space.

The output matrix has dimensions **4 × 5**:

```text
W_out =
[ 2.0   1.5   0.8  -1.0   0.1 ]
[-0.5   0.2   0.1   0.5  -0.2 ]
[ 1.8  -1.0   0.5  -2.0   0.0 ]
[ 0.4   0.1  -0.2   0.3   1.2 ]
```

Each column corresponds to one token in the vocabulary.

> **Note:** Some LLMs tie the output projection weights to the input embedding matrix. For this toy example, we treat the output projection independently so the mathematics is easier to follow.

---

## Step 1: Logit Generation

The Language Modeling Head projects the context vector into vocabulary space:

```text
z = h × W_out
```

The result is one **logit** for every possible next token.

A logit is a raw score. It is **not yet a probability**.

### Logit Calculation

#### `great`

```text
(1.2 × 2.0) + (-0.5 × -0.5) + (2.0 × 1.8) + (0.1 × 0.4)

= 2.4 + 0.25 + 3.6 + 0.04

= 6.29
```

#### `terrible`

```text
(1.2 × 1.5) + (-0.5 × 0.2) + (2.0 × -1.0) + (0.1 × 0.1)

= 1.8 - 0.1 - 2.0 + 0.01

= -0.29
```

#### `okay`

```text
(1.2 × 0.8) + (-0.5 × 0.1) + (2.0 × 0.5) + (0.1 × -0.2)

= 0.96 - 0.05 + 1.0 - 0.02

= 1.89
```

#### `banana`

```text
(1.2 × -1.0) + (-0.5 × 0.5) + (2.0 × -2.0) + (0.1 × 0.3)

= -1.2 - 0.25 - 4.0 + 0.03

= -5.42
```

#### `the`

```text
(1.2 × 0.1) + (-0.5 × -0.2) + (2.0 × 0.0) + (0.1 × 1.2)

= 0.12 + 0.1 + 0.0 + 0.12

= 0.34
```

### Raw Logits

```text
z = [6.29, -0.29, 1.89, -5.42, 0.34]
```

At this point, `great` has the highest score, but we still do **not** have probabilities.

---

## Step 2: Temperature Scaling

Temperature changes how concentrated or spread out the eventual probability distribution will be.

The scaled logits are calculated as:

```text
scaled_logit = logit / temperature
```

- **Temperature below 1:** makes the distribution sharper.
- **Temperature = 1:** leaves the logits unchanged.
- **Temperature above 1:** makes the distribution flatter.

### T = 0.7 vs T = 2.0

| Token | Raw Logit | T = 0.7 | T = 2.0 |
|---|---:|---:|---:|
| `great` | 6.29 | 8.986 | 3.145 |
| `okay` | 1.89 | 2.700 | 0.945 |
| `the` | 0.34 | 0.486 | 0.170 |
| `terrible` | -0.29 | -0.414 | -0.145 |
| `banana` | -5.42 | -7.743 | -2.710 |

Notice what happened:

- At **T = 0.7**, differences between the logits become larger.
- At **T = 2.0**, differences between the logits become smaller.

The effect becomes clearer after Softmax.

---

## Step 3: Probability Distribution with Softmax

Softmax converts the scaled logits into probabilities.

For each token:

```text
Probability(token)
=
exp(scaled_logit)
/
sum of exp(all scaled logits)
```

All probabilities add up to 100%.

### Probabilities at T = 0.7

The scaled logits are approximately:

```text
[8.9857, -0.4143, 2.7000, -7.7429, 0.4857]
```

The exponentials are approximately:

```text
exp(8.9857)  = 7988.15
exp(-0.4143) = 0.6608
exp(2.7000)  = 14.8797
exp(-7.7429) = 0.000434
exp(0.4857)  = 1.6253
```

Their sum is approximately:

```text
8005.31
```

So the probabilities are approximately:

| Token | Probability at T = 0.7 |
|---|---:|
| `great` | **99.786%** |
| `terrible` | **0.0083%** |
| `okay` | **0.1859%** |
| `banana` | **0.0000054%** |
| `the` | **0.0203%** |

`great` overwhelmingly dominates the distribution.

### What happens at T = 2.0?

Using the scaled logits:

```text
[3.145, -0.145, 0.945, -2.710, 0.170]
```

we get approximately:

| Token | T = 0.7 | T = 2.0 |
|---|---:|---:|
| `great` | **99.786%** | **83.197%** |
| `okay` | 0.186% | **9.218%** |
| `the` | 0.020% | **4.247%** |
| `terrible` | 0.0083% | **3.099%** |
| `banana` | 0.0000054% | **0.238%** |

This shows the effect of temperature clearly:

**Lower temperature → probability mass concentrates around the strongest candidate.**

**Higher temperature → probability mass spreads across more candidates.**

Temperature does not directly make a model "creative." It changes the probability distribution, which can result in more varied outputs when sampling is used.

---

## Step 4: Filtering with Top-K and Top-P

The T = 0.7 distribution above is extremely concentrated around `great`, so it is not very useful for demonstrating Top-K and Top-P.

For the following examples, we will use a **separate illustrative probability distribution** that is intentionally more balanced:

| Token | Probability |
|---|---:|
| `great` | 70.0% |
| `okay` | 18.0% |
| `terrible` | 8.0% |
| `the` | 3.5% |
| `banana` | 0.5% |

Total:

```text
70% + 18% + 8% + 3.5% + 0.5% = 100%
```

### Case A: Top-K, K = 2

Top-K asks:

> **"How many candidates should I keep?"**

With **K = 2**, we keep only the two highest-probability tokens:

```text
great
okay
```

The other three candidates are discarded.

Top-K is based on **rank**, not on a probability threshold.

---

### Case B: Top-P, P = 90%

Top-P asks:

> **"How much probability mass should I cover?"**

First, sort the candidates from highest to lowest probability:

| Token | Probability | Cumulative | Action |
|---|---:|---:|---|
| `great` | 70% | 70% | Keep |
| `okay` | 18% | 88% | Keep |
| `terrible` | 8% | 96% | **Keep & Stop** |
| `the` | 3.5% | 99.5% | Discard |
| `banana` | 0.5% | 100% | Discard |

With **P = 90%**, we keep:

```text
great
okay
terrible
```

Notice something important:

> **Top-P does not necessarily stop exactly at 90%. It stops at the first token that makes the cumulative probability reach or exceed 90%.**

Here, the cumulative probability jumps from 88% to 96%, so the final retained probability mass is 96%.

### Top-K vs Top-P

The easiest way to remember the difference:

**Top-K → How many candidates?**

**Top-P → How much probability mass?**

---

## Step 5: Re-Normalization and Final Sampling

Let's continue with the Top-P example.

The filtered set is:

| Token | Original Probability |
|---|---:|
| `great` | 70% |
| `okay` | 18% |
| `terrible` | 8% |

### 1. Sum of Filtered Probabilities

```text
70% + 18% + 8% = 96%
```

The remaining candidates were discarded, so the retained probabilities add up to only 96%.

Before sampling, we re-normalize them so they add up to 100%.

### 2. Re-Normalize

New probability:

```text
new probability = original probability / total retained probability
```

#### `great`

```text
70 / 96 = 72.92%
```

#### `okay`

```text
18 / 96 = 18.75%
```

#### `terrible`

```text
8 / 96 = 8.33%
```

Check:

```text
72.92% + 18.75% + 8.33% = 100%
```

So the final sampling distribution is:

| Token | Final Probability |
|---|---:|
| `great` | **72.92%** |
| `okay` | **18.75%** |
| `terrible` | **8.33%** |

---

### 3. Sampling

Sampling does **not** mean every remaining token has an equal chance.

Each token gets a chance according to its final probability.

We can visualize the distribution as intervals between 0 and 1:

```text
0.0000 ─────────────── 0.7292
          great

0.7292 ─────────────── 0.9167
          okay

0.9167 ─────────────── 1.0000
          terrible
```

Now draw a random number between 0 and 1.

Suppose:

```text
r = 0.82
```

Since:

```text
0.7292 ≤ 0.82 < 0.9167
```

the selected token is:

### **`okay`**

Notice that `great` was still the most likely choice, but `okay` and `terrible` still had a chance.

That is the key idea behind sampling.

---

## The Complete Pipeline

We can now connect everything together:

```text
Context Vector
      ↓
Language Modeling Head
      ↓
Raw Logits
      ↓
Temperature Scaling
      ↓
Scaled Logits
      ↓
Softmax
      ↓
Probability Distribution
      ↓
Top-K / Top-P Filtering
      ↓
Re-Normalization
      ↓
Sampling
      ↓
Next Token
      ↓
Context grows with the new token
      ↓
Repeat
```

The model does not generate the entire response in one shot.

It predicts a probability distribution for the **next token**, selects one token, adds it to the sequence, and then repeats the process.

So the complete idea is:

> **Logits tell us the model's raw preferences.**  
> **Temperature reshapes those preferences.**  
> **Softmax turns them into probabilities.**  
> **Top-K and Top-P narrow the candidate pool.**  
> **Re-normalization restores the probabilities to 100%.**  
> **Sampling chooses the next token.**

And then the whole process starts again — **one token at a time.**
