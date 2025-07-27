import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.preprocessing import OrdinalEncoder  # Uncomment if you prefer ordinal
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
# 1. Load the data
housing = pd.read_csv("housing.csv")

# 2. Create a stratified test set based on income category
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# Work on a copy of training data
housing = strat_train_set.copy()

# 3. Separate predictors and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# 4. Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. Pipelines
# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical pipeline
cat_pipeline = Pipeline([
    # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)

# housing_prepared is now a NumPy array ready for training
#print(housing_prepared.shape)
# 7. train the model

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_reg_preds= lin_reg.predict(housing_prepared) 
# lin_reg_rmse=root_mean_squared_error(housing_labels,lin_reg_preds)
lin_reg_rmses= -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error",cv=10)
# print(f"the root mean sqaured error for linear regression is {lin_reg_rmse}")
print(pd.Series(lin_reg_rmses).describe())
# Decision Tree
dec_reg = DecisionTreeRegressor(random_state=42)
dec_reg.fit(housing_prepared, housing_labels)
dec_reg_preds=dec_reg.predict(housing_prepared)

# dec_reg_rmse=root_mean_squared_error(housing_labels,dec_reg_preds)
dec_rmses= -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error",cv=10)
# print(f"the root mean sqaured error for decision tree reggresor is {dec_rmses}") 
print(pd.Series(dec_rmses).describe())
# Random Forest
random_forest_reg = RandomForestRegressor(random_state=42)
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_preds=random_forest_reg.predict(housing_prepared)
# random_forest_rmse=root_mean_squared_error(housing_labels,random_forest_preds)
random_forest_rmses= -cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error",cv=10)
# print(f"the root mean sqaured error for random forest reggresor is {random_forest_rmse}")
print(pd.Series(random_forest_rmses).describe())