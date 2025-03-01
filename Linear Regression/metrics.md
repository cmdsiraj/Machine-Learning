## 📌 Common Metrics in Linear Regression
These metrics measure how well the model fits the data by comparing **actual values (y)** with **predicted values (\(\hat{y}\))**.

### **1️⃣ Mean Absolute Error (MAE)**
```math
MAE = \frac{1}{n} \sum | y_i - \hat{y}_i |
```
- Measures the **average absolute difference** between actual and predicted values.  
- **Lower is better** → A smaller MAE means better predictions.  
- **Good for interpretability**, as it’s in the same unit as the target variable.  

### **2️⃣ Mean Squared Error (MSE)**
```math
MSE = \frac{1}{n} \sum ( y_i - \hat{y}_i )^2
```
- Penalizes **larger errors** more than MAE since it squares the differences.  
- **Lower is better**, but the unit is squared (not intuitive for interpretation).  

### **3️⃣ Root Mean Squared Error (RMSE)**
```math
RMSE = \sqrt{MSE}
```
- **Square root of MSE**, so it's in the same unit as the target variable.  
- **More sensitive to large errors** compared to MAE.  
- **Lower RMSE = better model performance.**  

### **4️⃣ Coefficient of Determination (\( R^2 \))**
```math
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
```
Where:  
- \( SS_{res} \) = Sum of squared residuals \( \sum ( y_i - \hat{y}_i )^2 \)  
- \( SS_{tot} \) = Total sum of squares \( \sum ( y_i - \bar{y} )^2 \)  

**Interpretation:**  
- \( R^2 \) measures how much of the variance in \( y \) is explained by \( x \).  
- **Range:** \( 0 \leq R^2 \leq 1 \)  
  - **\( R^2 = 1 \)** → Perfect fit (all points on the regression line).  
  - **\( R^2 = 0 \)** → Model explains nothing; worse than a horizontal line.  
  - **Negative \( R^2 \)** → Worse than just predicting the mean.  

### **5️⃣ Adjusted \( R^2 \) (For Multiple Regression)**
```math
Adjusted\ R^2 = 1 - (1 - R^2) \times \frac{n - 1}{n - p - 1}
```
Where \( n \) is the number of observations and \( p \) is the number of predictors.  
- **Adjusts for the number of independent variables.**  
- Prevents overfitting by penalizing unnecessary predictors.  

---

## 🚀 **Final Summary**
| **Metric** | **What it Measures** | **Good Value** |
|------------|---------------------|---------------|
| **MAE**  | Avg absolute error | Lower is better |
| **MSE**  | Avg squared error | Lower is better |
| **RMSE**  | More sensitive to large errors | Lower is better |
| **\( R^2 \)** | Proportion of variance explained | Closer to 1 |
| **Adjusted \( R^2 \)** | Corrected \( R^2 \) for multiple variables | Closer to 1 |

---

## **When to Use Which Metric?**
- **MAE** → Easy to interpret but treats all errors equally.  
- **MSE/RMSE** → Penalizes large errors more (good for catching big mistakes).  
- **\( R^2 \)** → Measures overall goodness of fit.  
- **Adjusted \( R^2 \)** → Better for **multiple regression** (avoids overfitting).  

---
