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
  1. **GridSearchCV**:
     - It tries all the combinations of the given paramenter ranges.
     - Computationaly expensive and time consuming.
     - **Example**:
       - Let's say we have 3 parameters and we have give 1000 values of each paramenter to test fot ***GridSearchCV***, then it will try out all the 1000 combinations.
  2. **RandomizeSearchCV**:
     - In this, it samples the subsets of paramater ranges and tests them.
     - So, it's computationaly inexpensive compared to `GridSearchCV` and less time taken.
> **When `RandomizeSearchCV` tests only subset of values, then how can it be sure that the best possible combinations is with in this subset?**<br>
> Well, it doesn't know that it is in this subset. It may not find the Global optimum, but it tries to get close to best possible values possible. it's just a **trade-off between time and performance**

| Common Strategy |
|---------------------|
| Use `RandomizedSearchCV` to find a number of random samples from a wide range of parameter values. Then use `GridSearchCV` to test small incremental changes around the best values found by `RandomizedSearchCV`. |

________________________________________________________________

# Ensembles
- Ensembles involve combining a number of weaker models into an ensemble. The end result is a stronger aggrigate model.
> **Why combining weak learners can help?**
> - A **weak learner** (say a shallow tree) has high bias and low variance.
> - One model like this performs poorly.
> - But if we train many such learners on **different data slices**, each model will make different **mistakes**.
> - **Mathematically:**
>   If each model has an error that’s independent, combining them can reduce overall error:
>   ![Variance of Mean](https://latex.codecogs.com/png.image?\dpi{150}Var(\bar{M})=\frac{1}{n^2}\sum_{i=1}^{n}Var(M_i)=\frac{\sigma^2}{n})

- Three Populor ensemble methods:
  - Bagging
  - Boosting
  - Stacking

## Bagging
- **High variance** models like decision trees can overfit.
- Bagging reduces this variance by training many models on different **bootstrapped samples**(randmonly sampled with replacement)
- Predictions are then **Averaged** (for regresssion) or **majority-voted** (for classification).
### **Working:**
  1. **Boostrap sampling:**
     - From a dataset of size `n`, generate `k` different datasets, each also of size `n` but sampled with replacement.
  2. **Train** a base learner (eg: decision tree) on each of these datasets.
  3. **Combine** predictions:
     - For regression: Average of output of each base learner.
       ![Var of Mean](https://latex.codecogs.com/png.image?\dpi{150}Var(\bar{M})=\frac{\sigma^2}{k}\quad\text{(if%20models%20are%20independent)})
     - For classification: Majority Vote.
### **Why does it reduce variance?**
  - Lets say each model is an unbaised estimator with variance σ².
  - The the average of `k` such models has vairance:

> **Random Forest (a bagging method)**<br>
> In addition to sampling rows, it **randomly samples features when splitting a node.**<br>
> This makes individual trees **less correlated**, which means more variance reduction when averaged.

### **Weakness:**
  - Bagging doesn't reduce bias.
  - If base models are baised, then the ensemble remains biased.

## Boosting
- **Build models sequentially, where each new model focuses on the mistakes of the previous ones.**
- This not only just combines models, but each model gets smarter by **learning from the errors** of the last.
- The goal is to reduace **bias**
### Working
- We want to predict y from features X.
- We build a series of weak models M1, M2,...MT.
- Each model adds to final prediction:
  -
  - α: learning rate (controls how much you trust the new model)
- The final prediction is the sum of all weak learners.
  > 1. Predict
  > 2. Calculate the **residuals** or **loss gradianet**
  > 3. Train next model to fit those residuals
  > 4. Add to the previous model
  > <br>This is why boosting is also called "**Additive Modeling.**"
- Two types of Boosting algorithms:
#### AdaBoost Algorithm
- **Goal:** Build an ensemble by giving more importance to the examples that were misclassified.
- **Algorithm:**
  1. **Start:**
     - Assign **equal weight** to every data point (all are equally important at starting).
  2. **Repeat for T rounds** (number of weak learners):
     1. Train a weak model (e.g., a decision stump) on the **weighted data**.
     2. Check how many examples it got wrong.
     3. If it’s better than random, keep it; otherwise, skip or stop.
     4. Increase the weights of the **wrongly predicted** examples.
     5. Decrease the weights of the **correctly predicted** examples.
     6. Normalize weights so they sum to 1.
     7. Store this model with its importance score.
  3. **Final Prediction:**
     - Combine all the stored models.
     - Models that did well have more say in the final output (weighted majority vote).
#### Gradient Boosting (GBM) Algorithm
- **Goal**: Build an ensemble by sequentially correcting errors made by previous models.
- **Algorithm:**
  1. **Start:**
     - Create a simple model that makes a basic guess (e.g., mean of target values).
  2. **Repeate for T rounds:**
     1. Calculate how much each prediction is off (the **residuals** or errors).
     2. Train a new weak model to predict these **errors**.
     3. Add this new model to the previous one. it helps correct the mistake.\
     4. Scale its effect with a small **learning rate** (so we don't over-correct).
  3. **Final Prediction:**
     - Add up all the small corrections from all the models.
     - That sum gives the final prediction.

#### XGBoost (eXtreme Gradient Boosting)
- It's an optimized, scalable version of gradient boosting that includes:
  - Regularization
  - Efficent handling of sparse data
  - Parallel tree building
  - Pruning
  - Early stopping
  - Out-of-core trainig (for large datasets).
- It builds trees sequentially like GBM, but smarter and faster.

## Stacking
- **Core Idea**
  - Combines predictions from multiple **different type** of learners.
  - Uses a meta-learner (another model) to learn the best way to combine the base learners' predictions.
- **Analogy**: A manager learning how to best weigh advice from diverse experts (e.g., finance, engineering, marketing) to make a final decision.
### Working
- **Level 0**:
  - Train several diverse models (e.g., KNN, SVM, RandomForest, Logistic Regression) on the training data.
  - **Crusially**: Generate predictions from these base models on the training data itself, typically using cross-validation ("out-of-fold" predictions) to create meta-features. This prevents data leakage.
- **Level 1**:
  - Train a (usually simple) meta-learner model (e.g., Logistic Regression, Ridge, simple Tree).
  - **Input**: The meta-features (predictions from Level 0 models).
  - **Output**: The original target variable.
- **Prediction**: For new data, get predictions from all base models, feed these predictions into the trained meta-learner to get the final output.
- Goal: Leverage the unique strengths and perspectives of different algorithms to potentially achieve better performance than any single model or simpler ensembles (like Bagging/Boosting).
- Often uses strong, well-performing base models.
_____
> ***Ensembles reduce Errors by reducing bias (Boosting), variance (Bagging) or combining diverse models (stacking)***.
__________________
## Pros and Cons of each ensemble methods
| Method      | Pros                                                                 | Cons                                                                 |
|-------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| **Bagging** | - Reduces variance                                                   | - Does not reduce bias                                               |
| (e.g., Random Forest) | - Works well with high-variance models (like decision trees)      | - Trees can still be large and overfit individually                  |
|             | - Easy to parallelize (independent models)                           | - Less effective on biased models                                    |
|             | - Robust to overfitting                                              |                                                                      |
|             |                                                                      |                                                                      |
| **Boosting**| - Reduces bias effectively                                           | - Can overfit if not tuned properly                                  |
| (e.g., AdaBoost, GBM) | - Builds strong learners from weak ones                          | - Sequential process: slower training                                |
|             | - Highly accurate                                                    | - Sensitive to noisy data                                           |
|             | - Supports custom loss functions (in GBM)                            | - Harder to parallelize (sequential dependency)                      |
|             |                                                                      |                                                                      |
| **XGBoost** | - Regularized boosting: controls overfitting                         | - More complex to tune (many hyperparameters)                        |
|             | - Fast and efficient (parallelized split finding)                   | - Still sequential at tree level                                     |
|             | - Handles missing values natively                                    |                                                                      |
|             | - Excellent performance in competitions                              |                                                                      |
|             |                                                                      |                                                                      |
| **Stacking**| - Combines strengths of different model types                        | - Risk of overfitting if meta-learner is not well chosen             |
|             | - Very flexible and powerful                                         | - Requires careful data partitioning (e.g., k-fold for base models)  |
|             | - Can outperform single-type ensembles                               | - More computationally expensive                                     |

----------------------------------
## Notes
 - When working with trees, we no need to convert category features into  `OneHot Encoded` values.
   - **Why?**
     - Trees don't assume any relation between feature values (like 0<5<9).
     - They just split nodes based on yes/no. so order of the categorical variables doesn't matter.
     - Also, while Working with `RandomForest`, having `OneHotEncoded` features will confuse the model making it complex and reduced in performance because the model doesn't know that these `OneHotEncoed` features are related (i.e., belong to one feature). so we get problem when taking the subset of features while splitting.
