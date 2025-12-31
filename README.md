# Titanic-Survival-Analysis
A Random Forest model to predict Titanic survival using Feature Engineering (FamilySize)
# Titanic Survival Prediction
This project uses a **Random Forest Classifier** to predict survival on the Titanic.

### Why my model is unique:
- **Feature Engineering:** I created a `family_size` column to capture how group dynamics affected survival.
- **Handling Imbalance:** I used `class_weight='balanced'` to improve the model's ability to find survivors.
- **Experimental Constraints:** I intentionally removed the `Sex` column to see how much the model could learn from Class and Fare alone.

### Results:
- **Accuracy:** 61.90%
- **Recall:** 44.12% (Improved from 26% by using balanced weights)
