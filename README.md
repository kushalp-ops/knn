# K-Nearest Neighbors (KNN) Iris Classification

A Machine Learning project demonstrating the implementation of the **K-Nearest Neighbors (KNN)** algorithm using `scikit-learn` on the classic **Iris dataset**.

---

## 📌 Theoretical Overview

K-Nearest Neighbors (KNN) is a non-parametric, **lazy learning algorithm** that classifies new instances based on feature similarity to stored training examples.

### Key Steps:
1. **Memorize**: Store the entire training dataset during initialization.
2. **Calculate Distance**: Compute distance metrics (e.g., Euclidean distance via Minkowski distance with $p=2$) between the query data point and all stored training samples.
3. **Find Neighbors**: Identify the top $k$ nearest neighbors based on the calculated distances.
4. **Majority Voting**: Assign the class label based on majority consensus among the $k$ neighbors.

> **Note**: Choose $k$ as an **odd number** (e.g., $k=3$) to prevent tie votes in binary/multi-class classification.

---

## 📊 Dataset Information

The project uses the **Iris Flower Dataset** loaded directly via `sklearn.datasets.load_iris`:
* **Total Samples**: 150 (50 per class)
* **Features** (4 continuous numeric attributes):
  * Sepal length (cm)
  * Sepal width (cm)
  * Petal length (cm)
  * Petal width (cm)
* **Target Classes**:
  * `0`: Setosa
  * `1`: Versicolor
  * `2`: Virginica

---

## 🛠️ Implementation & Workflow

### Dependencies
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
