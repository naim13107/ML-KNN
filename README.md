#  MNIST Digit Classification using PCA + KNN

This project builds a complete machine learning pipeline to classify
handwritten digits from the **MNIST dataset**

------------------------------------------------------------------------

## Project Highlights

-   Load MNIST CSV dataset
-   Scale features using **MinMaxScaler**
-   Reduce dimensionality using **PCA (50 components)**
-   Train a **K-Nearest Neighbors (KNN)** classifier
-   Use **GridSearchCV** for hyperparameter tuning
-   Visualize PCA variance & confusion matrix
-   Achieve high accuracy on MNIST test set

------------------------------------------------------------------------

##  Project Structure
    ├── Data    
    ├── README.md
    └── mnist_knn_pca.ipynb                       

------------------------------------------------------------------------

## Workflow Summary

### **1. Load Dataset**

Reads the MNIST CSV dataset consisting of 784 features (flattened 28×28
pixels) and a label column.

### **2. Preprocessing**

-   Feature scaling with `MinMaxScaler`
-   Dimensionality reduction using **PCA (n_components=50)**\
    → Achieves large reduction while retaining most variance.

### **3. Visualization**

-   Display sample MNIST digit
-   PCA 2D scatter plot
-   PCA explained variance plot

### **4. Model Training**

Uses **K-Nearest Neighbors (KNN)**
Hyperparameters tuned using **GridSearchCV**: - `n_neighbors` from 1 to
20
- `weights`: uniform or distance

### **5. Evaluation**

-   Best hyperparameters
-   Test accuracy score
-   Classification report
-   Confusion matrix heatmap

------------------------------------------------------------------------

## Results

-   The optimized KNN model performs strongly on the MNIST test set.
-   PCA reduces dimensionality by over **90%** while maintaining high
    accuracy.

------------------------------------------------------------------------

## How to Run This Project

### **1. Clone the Repository**

``` bash
git clone https://github.com/naim13107/mnist-pca-knn.git
cd mnist-pca-knn
```
### ** 2. Download the datasets **

``` bash
Link given in Data/data_availivility.txt
```

### **3. Run the Notebook**

``` bash
jupyter notebook mnist_portfolio_ready.ipynb
```

------------------------------------------------------------------------

## Requirements

-   numpy
-   pandas
-   matplotlib
-   seaborn
-   scikit-learn

------------------------------------------------------------------------


## Author

**Md.Naim-Ul-Haque**\
BS in Applied Mathematics --- University of Dhaka\
Machine Learning & Data Science Enthusiast

------------------------------------------------------------------------

##  Support

If you like this project, consider giving the repository a **star**!
