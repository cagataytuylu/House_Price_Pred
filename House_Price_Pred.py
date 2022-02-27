######################################
# House Price Prediction
######################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling
# 4. Feature Selection
# 5. Hyperparameter Optimization with Selected Features
# 6. Sonuçların Yüklenmesi

import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import numpy as np
from helpers.data_prep import *
from helpers.eda import *

# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

######################################
# Exploratory Data Analysis
######################################

train = pd.read_csv("datasets/house_prices/train.csv")
test = pd.read_csv("datasets/house_prices/test.csv")
df = train.reset_index(drop=True)
df.head()


check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)



##################
# Categorical Variable Analysis
##################


for col in cat_cols:
    cat_summary(df, col)

for col in cat_but_car:
    cat_summary(df, col)



##################
# Numerical Variable Analysis
##################


df[num_cols].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.99]).T


# for col in num_cols:
#     num_summary(df, col, plot=True)



##################
# Target Analysis
##################


df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T


def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations

low_corrs, high_corrs = find_correlation(df, num_cols)




##################
# Rare Encoding
##################

rare_analyser(df, "SalePrice", cat_cols)


df = rare_encoder(df, 0.01, cat_cols)

rare_analyser(df, "SalePrice", cat_cols)


useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis=None))]

cat_cols = [col for col in cat_cols if col not in useless_cols]


for col in useless_cols:
    df.drop(col, axis=1, inplace=True)


rare_analyser(df, "SalePrice", cat_cols)



##################
# Label Encoding & One-Hot Encoding
##################

cat_cols = cat_cols + cat_but_car


df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)


cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "SalePrice", cat_cols)


useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.01).any(axis=None)]

df[useless_cols_new].head()


for col in useless_cols_new:
    cat_summary(df, col)


rare_analyser(df, "SalePrice", useless_cols_new)


##################
# Missing Values
##################

missing_values_table(df)

test.shape

missing_values_table(train)


na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and "SalePrice" not in col]

df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)


##################
# Outliers
##################

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.01, q3=0.99))

# for col in num_cols:
#     replace_with_thresholds(df, col, q1=0.01, q3=0.99)
######################################
# Modeling
######################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

# y = train_df["SalePrice"]
y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)


##################
# Base Models
##################

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())
          # ,("CatBoost", CatBoostRegressor(verbose=False))
          ]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")



##################
# Hyperparameter Optimization
##################


lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model,
                                        X, y, cv=10, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500, 5000, 10000],
               "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1]}


lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(X, y)



final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)


rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))



# rmse
# Out[26]: 0.12698595788807257


#######################################
# Feature Selection
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(final_model, X)


plot_importance(final_model, X, 20)


X.shape

feature_imp = pd.DataFrame({'Value': final_model.feature_importances_, 'Feature': X.columns})


# num_summary(feature_imp, "Value", True)


zero_imp_cols = feature_imp[feature_imp["Value"] < 1]["Feature"].values


selected_cols = [col for col in X.columns if col not in zero_imp_cols]
len(selected_cols)


######################################
# Data Preprocessing & Feature Engineering
######################################
feature_imp.sort_values(by="Value", ascending=False).head(20)

imp_feaures = feature_imp.sort_values(by="Value", ascending=False).head(20)

low_corr, high_corr = find_correlation(df, imp_feaures["Feature"])

high_corr   # check these features and try to produce new features

df["Area"] = df["GrLivArea"] / df["LotArea"]
df["Basement_score"] = df["BsmtUnfSF"] / df["TotalBsmtSF"]
df["first_fl"] = df["1stFlrSF"] / df["GrLivArea"]
df["basement2"] = df["TotalBsmtSF"] / df["GrLivArea"]
df['New_TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df["NEW_TotalFeet"] = df["GrLivArea"] + df["TotalBsmtSF"]
df["New_Relation"] = (df["OpenPorchSF"] + df["EnclosedPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]) / (df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'])

zero_imp_cols = feature_imp[feature_imp["Value"] < 1]["Feature"].values


selected_cols = [col for col in X.columns if col not in zero_imp_cols]

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

# y = train_df["SalePrice"]
y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)





#######################################
# Hyperparameter Optimization with Selected Features
#######################################


lgbm_model = LGBMRegressor(random_state=46)

lgbm_params = {"learning_rate": [0.01],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.05, 0.2],
               "num_leaves": [19],
               "min_split_gain": [0,1,2]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X[selected_cols], y)


final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X[selected_cols], y)
lgbm_model.get_params()
lgbm_gs_best.best_params_
rmse = np.mean(np.sqrt(-cross_val_score(final_model, X[selected_cols], y, cv=5, scoring="neg_mean_squared_error")))
rmse


# Out[first]: 0.12800403810825906
# Out[second]: 0.12487168974346649

#######################################
# Loading the Results to Kaggle
#######################################


submission_df = pd.DataFrame()

submission_df['Id'] = test_df["Id"]

y_pred_sub = final_model.predict(test_df[selected_cols])


y_pred_sub = np.expm1(y_pred_sub)

submission_df['SalePrice'] = y_pred_sub

submission_df.to_csv('submission.csv', index=False)


# kaggle score: 0.12677























