# Ensemble Methods
#### Revisiting Decision Trees:
- Desicion Trees are supervised learning models.
- Two types of trees:
  - Classification Trees (Out-come is classification)
  - Regression Tress (Out come is Continues)
_____
## Problems with Decision Trees:
- We have two potential problems with trees:
  - **Overfitting**: They tend to overfit on data, reducing the performance of the model on future data.<br>
    (A decision tree can split deeply to perfectly fit the training data -- memorizing noise and specific patterns in data.)
  - **[Variance error](https://github.com/cmdsiraj/Machine-Learning/blob/main/Misc.md#variance-error)**: Small changes in data can yield a big change in resulting tree.<br>
    (Since decision trees tend to overfit, a slighlty different dataset can produce different splits.)    

- Model errors (Error's in the models prediction) can be segmented into two parts: Errors we can correct and Errors we cannot.<br>
      `Model Error = Reducible Error + Irreducible Error`
- **Reducible error** is further segmented to two types:
    - **Bias Error**
    - **Variance Error**
- **Irreducible Errors** are the errors that cannot be reduced by any algorithm choice. These errors can be introduced through ***framing/context*** of the problem like ***Unkown/unrecorded variables, Measurment errors, etc.,***.

### Model Bias Error
- It measures how much the model assumes the underlying relationship between input and outputs.
- **Low Bias**
  - The model makes fewer assumptions about the relationship.
  - **More Flexible**
  - Example: Decision trees, K-NN, Neural Networks.
- **High Bias**
  - The model makes **Strong Assusmptions.** (Assuming Linear Relationship).
  - **Less Flexible**
  - Example: Linear Regression.
- A model with **high bias is helpful** when the **strong assumptions matches the true relationships**, else the model performs very poorly.
### Variance Error
- This is the amount.magnitude of change in predictions if different data is used.
- **Low Varince**:
  - Changes in training data only results in small changes in predictions.
  - **Examples**: Linear Regresssion, Linear Discriminant Analysis and Logistic Regression.
- **High Variance**:
  - Changes in training data will result in large changes in predictions.
  - **Examples**: Decision Trees, K-NN and Support Vector Machines.
- High variance models will perfectly match the relationships between input and target in the training data, but will often perform poorly on new data. (Overfitting)

### Bias and Variance Tradeoff:
- **Increasing Flexibility (low bias)** will **Increase variance**.
- **Simpler models (high bias)** have **lower variance**.<br>
  ![Bias-Variance](https://latex.codecogs.com/png.image?\dpi{150}Bias\propto\frac{1}{\mathrm{Variance}})

### Managing Bias and Variance
- **Limiting Bias**
  - Limiting Bias means we are making our model more flexible by limiting the assumptions.
  - The more assumptions we make about the relation, the more we introduce the bias (this isn't necessarily bad, but can be if applicable).
  - What is important is the performance of the model on the data we actually have.
- **Limiting Variance**
  - We can limit by using the techniques:
    - Pruning/constraining your model
    - Bagging and Resampling
- Getting more data will help manage the problem of variance but will not impact the bias. Bias can only be changed by altering the modeling techniques.

### Bias and Variance in Decision Trees:
- **Low Bias**
  - Not impeded by strong assumptions/constraints
  - Will fit/model complex relationships.
- **High Variance**
  - Small changes in the training set will results in large changes in predictions/model produced.
______________
## Addressing the problems with Decision Trees:
- The problem of variance for Decision Trees can be addressed in two ways:
  1. **Pruning**
  2. **Ensemble Techniques**

### 1. Pruning
- So basically, decreasing varaince means priventing the DT from overfitting.
- In pruning, we can prevent the tree from overfitting by **removing the parts of the model that don't help much, making the model simpler**.
- Examples:
  - Limiting the depth of the Tree.
  - Setting the MIn samples per leaf.
  - Min Split Size.
#### How to do pruning in Skiti-learn (****)
- We have many parameters that can limit the Decision Tree structure (Prune) and each parameter can take different values (so a lot of combinations to try on).
- We can find the best match parameter values for our problem *dataset) by using some useful methods which are provided by the skiti-learn.
  1. 
