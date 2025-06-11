
## Step-by-Step Design Choices

### 1. Outlier Treatment for Age (tukey_age)
- **Transformer:** `CustomTukeyTransformer(target_column='Age', fence='outer')`  
- **Design Choice:** Outer fence clipping for extreme outliers  
- **Rationale:** Age may include rare but valid values (e.g., non-traditional students). The outer Tukey fence removes only the most extreme values without distorting the overall distribution.

### 2. Outlier Treatment for Study Satisfaction (tukey_study_satisfaction)
- **Transformer:** `CustomTukeyTransformer(target_column='Study Satisfaction', fence='outer')`  
- **Design Choice:** Outlier clipping to limit extreme satisfaction scores  
- **Rationale:** Unusual numeric responses (e.g., very high or very low) can bias the scaling process. Tukey clipping normalizes the range while preserving central tendencies.

### 3. Outlier Treatment for Work/Study Hours (tukey_work_hours)
- **Transformer:** `CustomTukeyTransformer(target_column='Work/Study Hours', fence='outer')`  
- **Design Choice:** Apply Tukey outer fence to limit implausibly high/low work hours  
- **Rationale:** Extreme reported hours (e.g., 0 or 100+) can skew the model; this step softens the effect of such outliers.

### 4. Age Scaling (scale_age)
- **Transformer:** `CustomRobustTransformer(column='Age')`  
- **Design Choice:** Apply robust scaling using median and IQR  
- **Rationale:** Age distribution may not be normal and may still contain mild outliers. RobustScaler handles this better than standard z-score methods.

### 5. Study Satisfaction Scaling (scale_study_satisfaction)
- **Transformer:** `CustomRobustTransformer(column='Study Satisfaction')`  
- **Design Choice:** Scale using IQR and median  
- **Rationale:** Prevents extreme satisfaction scores from dominating the range during modeling, while preserving interpretability.

### 6. Work/Study Hours Scaling (scale_work_hours)
- **Transformer:** `CustomRobustTransformer(column='Work/Study Hours')`  
- **Design Choice:** Scale using robust statistics  
- **Rationale:** Ensures comparability across features for later distance-based methods like KNN imputation.

### 7. Suicidal Thought Mapping (map_suicidal)
- **Transformer:** `CustomMappingTransformer('Have you ever had suicidal thoughts ?', {'Yes': 1, 'No': 0})`  
- **Design Choice:** Binary mapping of yes/no values  
- **Rationale:** Simple conversion to 0/1 format enables numeric processing and modeling while preserving meaning.

### 8. Target Encoding for Sleep Duration (target_sleep)
- **Transformer:** `CustomTargetTransformer(col='Sleep Duration', smoothing=10)`  
- **Design Choice:** Target encoding with smoothing  
- **Rationale:** Replaces raw categories with smoothed probability of depression. Smoothing (10) balances category-specific and overall depression rates, avoiding overfitting from rare categories.

### 9. Target Encoding for Dietary Habits (target_diet)
- **Transformer:** `CustomTargetTransformer(col='Dietary Habits', smoothing=10)`  
- **Design Choice:** Target encoding for dietary regularity  
- **Rationale:** Models the relationship between dietary patterns and depression likelihood in a smoothed, statistically meaningful way.

### 10. Target Encoding for Degree (target_degree)
- **Transformer:** `CustomTargetTransformer(col='Degree', smoothing=10)`  
- **Design Choice:** Encodes degree level based on target correlation  
- **Rationale:** Enables the model to learn subtle differences between education levels and depression risk, without inflating feature space.

### 11. Target Encoding for Financial Stress (target_fin_stress)
- **Transformer:** `CustomTargetTransformer(col='Financial Stress', smoothing=10)`  
- **Design Choice:** Encodes categorical financial stress with respect to depression label  
- **Rationale:** Captures the influence of perceived financial hardship on depression risk using statistically robust encoding.

### 12. Imputation (impute)
- **Transformer:** `CustomKNNTransformer(n_neighbors=5)`  
- **Design Choice:** Use KNN imputation to fill missing values  
- **Rationale:** Leverages similarity across fully preprocessed features to estimate missing values more effectively than mean/median imputation.  
  k=5 offers a good tradeoff between variance and bias.

## Pipeline Execution Order Rationale
1. **Outlier clipping** occurs first to prevent extreme values from distorting the next stages.
2. **Robust scaling** then normalizes numeric ranges, making features comparable and improving downstream modeling.
3. **Binary mapping** standardizes yes/no responses to numeric format.
4. **Target encoding** transforms categorical columns based on correlation with the depression target, capturing useful signal.
5. **Imputation** is last, ensuring that all prior transformations are applied before estimating missing values using KNN.

## Performance Considerations
- **Robust scaling** is preferred over standard scaling due to its resistance to outliers.
- **Target encoding** is used instead of one-hot encoding to reduce dimensionality and capture feature-target relationships directly.
- **KNN imputation** improves estimation of missing values by considering feature similarity in a multidimensional space.
