# 📌 Linear Regression: Quick Review Notes

## 1️⃣ What is Linear Regression?
Linear Regression is a statistical model that finds the relationship between an **independent variable (X)** and a **dependent variable (Y)** using a straight-line equation:

```math
Y = β_0 + β_1X + ε
```

Where:
- **β₀** = Intercept
- **β₁** = Slope (coefficient)
- **ε** = Error term (residuals)

---

## 2️⃣ Assumptions of Linear Regression
For the model to be valid, the following **5 assumptions** must hold:

### **1. Linearity**
✅ **The relationship between X and Y must be linear**.
- **Check:** Scatter plot or residual plot (should not show a curve).
- **Fix if violated:** Try **polynomial regression** or **log transformation**.

### **2. Independence of Errors (No Autocorrelation)**
✅ **Residuals should not be correlated with each other**.
- **Check:** Durbin-Watson test (ideal value ≈ 2).
- **Fix if violated:** Use **time-series models** like ARIMA, add **lag variables**.

### **3. Homoscedasticity (Constant Variance of Residuals)**
✅ **The spread of residuals should be uniform across all values of X**.
- **Check:** Residuals vs. Predicted plot (should not show a funnel shape).
- **Fix if violated:** Apply **log transformation** or **weighted regression**.

### **4. Normality of Residuals**
✅ **Residuals should be normally distributed** for valid hypothesis testing.
- **Check:** Histogram, Q-Q plot, or Shapiro-Wilk test.
- **Fix if violated:** Use **log transformation** or **robust regression**.

### **5. No Perfect Multicollinearity**
✅ **Independent variables should not be highly correlated with each other**.
- **Check:** **VIF (Variance Inflation Factor) > 5** suggests multicollinearity.
- **Fix if violated:** Remove highly correlated variables, use **PCA (Principal Component Analysis)**.

---

## 3️⃣ Residual Plot Interpretation
The **residual plot** helps diagnose model issues:

| **Pattern in Residual Plot** | **What It Means** | **Solution** |
|----------------------|----------------------|--------------|
| ✅ **Random spread around zero** | Model is correct | No action needed |
| ❌ **U-shaped or curve pattern** | Non-linearity | Use polynomial terms or transformations |
| ❌ **Funnel shape (increasing spread)** | Heteroscedasticity | Use log transformation or weighted regression |
| ❌ **Patterned residuals (waves/trends)** | Autocorrelation | Use time-series models (ARIMA) |
| ❌ **Few extreme points far from others** | Outliers | Identify and investigate (Cook’s distance, leverage statistics) |

---

## 4️⃣ Key Takeaways
✔ **If assumptions hold:** The model is valid, predictions are reliable.
❌ **If violated:** Model may give incorrect results; check and fix using residual analysis.

💡 **Quick Tip:** Always check **scatter plots, residual plots, and VIF values** before finalizing your model.

---
