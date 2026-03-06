

**Fluid Dynamics based image representation**

This project explores a new way to represent images by simulating **fluid flow over pixel structures**.
Instead of directly feeding pixels into neural networks, images are converted into **fluid disturbance patterns** using a Navier–Stokes simulation.

The resulting **vorticity fingerprints** are then used as features for classification.

This project investigates whether **physics-driven representations** can capture meaningful shape information.

---

# Overview

Traditional computer vision pipelines rely on deep neural networks to learn features directly from pixel data.

This project proposes a different approach:

```
Image
 ↓
Absorption Field (pixels act as obstacles)
 ↓
Fluid Flow Simulation (Navier–Stokes)
 ↓
Vorticity Field
 ↓
Spectral Signature
 ↓
Classifier (SVM / MLP)
```

The key idea:

> Objects create **unique disturbance patterns in a flowing medium**, and these patterns can serve as signatures for recognition.

---

# Method

### 1. Image → Flow Field

Each image is interpreted as an **absorption map** that interacts with a fluid flow.

* Bright pixels absorb more flow
* Dark pixels allow flow to pass

A **finite difference Navier–Stokes solver** simulates the flow field.

---

### 2. Vorticity Extraction

From the velocity field `(u, v)` we compute **vorticity**:

```
ω = ∂v/∂x − ∂u/∂y
```

The vorticity field represents **rotational disturbances** created by the shape.

---

# Dataset

Experiments were performed on:

**MNIST handwritten digits**

```
10 classes
28×28 grayscale images
```

Flow simulations were run for each image to produce vorticity fingerprints.

---

# Experimental Results

## A. Extreme Data Scarcity

Training with **very limited data**.

| Samples per Class | CNN Accuracy | Physics + SVM Accuracy |
| ----------------- | ------------ | ---------------------- |
| 1                 | 52.83%       | 42.34%                 |
| 5                 | 68.58%       | 58.13%                 |
| 20                | 81.75%       | 72.68%                 |

Even with **only 1 sample per class**, the physics-based representation performs far above random guessing (10%).

---

## B. Noise Robustness

Models trained on **clean data (200 samples per class)** and tested under noise.

| Noise Level | CNN Accuracy | Physics + SVM Accuracy |
| ----------- | ------------ | ---------------------- |
| 0.0         | 95.88%       | 89.28%                 |
| 0.3         | 89.13%       | ~85%                   |

Fluid simulation acts as a **natural smoothing process**, making the representation robust to noise.

---

# Representation Stability

Similarity experiments were conducted to evaluate representation consistency.

| Experiment                          | Similarity |
| ----------------------------------- | ---------- |
| Two different images of digit **5** | **0.9258** |
| Digit **5 vs 3**                    | **0.9148** |
| Digit **5 rotated by 15°**          | **0.9975** |
| Spectral similarity                 | **0.9646** |

These results suggest the representation captures **shape geometry rather than raw pixel patterns**.

---

# Example Pipeline

1. Input digit image
2. Fluid flow simulation
3. Velocity field computation
4. Vorticity heatmap generation
5. Spectral feature extraction
6. Classification

Visual outputs include:

* Flow velocity maps
* Vorticity fingerprints
* Spectral signatures

---


# Research Motivation

Most images are encoded as pixel data, this project explores a novel way to represent images using Fluid Dynamics 
Simplest analogy to understand this is how a water flows if you pour it on the image

---



Independent research project exploring **physics-based representation learning for vision**.
