## 📊 Step-by-Step Design Choices — Student Depression Pipeline

### 1. Outlier Treatment for Age (`tukey_age`)
**Transformer**: `CustomTukeyTransformer(target_column='Age', fence='outer')`  
**Design Choice**: Uses Tukey's outer fence to clip extreme outliers.  
**Rationale**:  
- Ensures extreme ages (e.g., <16 or >40 in this student set) don’t distort downstream scaling.  
- Outer fence is conservative and preserves most of the real distribution.

---

### 2. Outlier Treatment for Study Satisfaction (`tukey_study_satisfaction`)
**Transformer**: `CustomTukeyTransformer(target_column='Study Satisfaction', fence='outer')`  
**Design Choice**: Clips unusually low or high satisfaction scores.  
**Rationale**:  
- Scores are on a fixed scale (e.g., 1–5), but extreme values may be noisy.  
- Keeps the scale robust for modeling.

---

### 3. Outlier Treatment for Work/Study Hours (`tukey_work_hours`)
**Transformer**: `CustomTukeyTransformer(target_column='Work/Study Hours', fence='outer')`  
**Design Choice**: Clipping of extreme weekly workload hours.  
**Rationale**:  
- Some students report >60 hours/week, which can skew the distribution.  
- Tukey handles these safely before scaling.

---

### 4. Age Scaling (`scale_age`)
**Transformer**: `CustomRobustTransformer(column='Age')`  
**Design Choice**: RobustScaler using median and IQR.  
**Rationale**:  
- Resistant to outliers remaining after clipping.  
- Age is continuous and skewed in student data.

---

### 5. Study Satisfaction Scaling (`scale_study_satisfaction`)
**Transformer**: `CustomRobustTransformer(column='Study Satisfaction')`  
**Design Choice**: Same approach as with Age.  
**Rationale**:  
- Ensures fairness in downstream models using this feature.  
- Preserves ordinal nature of satisfaction.

---

### 6. Work/Study Hours Scaling (`scale_work_hours`)
**Transformer**: `CustomRobustTransformer(column='Work/Study Hours')`  
**Design Choice**: Normalize workload after clipping.  
**Rationale**:  
- Workload affects depression and varies heavily — robust scaling ensures stability.

---

### 7. Suicidal Thoughts Mapping (`map_suicidal`)
**Transformer**: `CustomMappingTransformer('Have you ever had suicidal thoughts ?', {'Yes': 1, 'No': 0})`  
**Design Choice**: Binary categorical mapping.  
**Rationale**:  
- Keeps a vital binary feature compact and interpretable.  
- Avoids creating unnecessary one-hot vectors for a Yes/No column.

---

### 8. Sleep Duration Encoding (`ohe_sleep`)
**Transformer**: `CustomOHETransformer('Sleep Duration')`  
**Design Choice**: One-hot encoding of nominal sleep categories.  
**Rationale**:  
- No clear ordinal relationship among `"5-6 hours"`, `"More than 8 hours"`, etc.  
- OHE handles this cleanly and supports interpretability.

---

### 9. Dietary Habits Encoding (`ohe_diet`)
**Transformer**: `CustomOHETransformer('Dietary Habits')`  
**Design Choice**: One-hot encoding of eating habits.  
**Rationale**:  
- Categories like `"Healthy"`, `"Moderate"`, `"Unhealthy"` are subjective.  
- OHE avoids forcing a false order on them.

---

### 10. Degree Encoding (`ohe_degree`)
**Transformer**: `CustomOHETransformer('Degree')`  
**Design Choice**: Nominal degree categories encoded as binary flags.  
**Rationale**:  
- Degrees are not ordinal (e.g., `"BSc"` vs. `"MSc"`), so OHE is safer.  
- Ensures flexibility for rare degrees.

---

### 11. Financial Stress Encoding (`ohe_fin_stress`)
**Transformer**: `CustomOHETransformer('Financial Stress')`  
**Design Choice**: Converts financial stress levels into indicator columns.  
**Rationale**:  
- Categories may be `"None"`, `"Moderate"`, `"High"` but unclear order.  
- OHE avoids assumptions about severity ranking.

---

### 12. Imputation (`impute`)
**Transformer**: `CustomKNNTransformer(n_neighbors=5)`  
**Design Choice**: KNN-based missing value imputation.  
**Rationale**:  
- Leverages feature similarity to estimate missing values.  
- `k=5` balances noise and accuracy.  
- More robust than mean or mode imputation for behavioral data.

---

## 📈 Pipeline Execution Order Rationale

- Outlier clipping is performed first to avoid skewing scaling.
- Scaling comes next to prepare numeric features for KNN imputation.
- Binary and one-hot encoding is applied after numeric prep.
- Imputation is performed last so all features are transformed before filling gaps.
- This order ensures consistency and protects against data leakage.

---

## ⚙️ Performance Considerations

- **RobustScaler** ensures skewed distributions (like workload) are centered without overreacting to outliers.
- **OHE** is safer than ordinal encoding for ambiguous categories.
- **KNN imputation** preserves feature relationships better than dropping rows or using static averages.
