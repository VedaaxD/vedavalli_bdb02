from numpy.ma.core import indices
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
california_housing=fetch_california_housing(as_frame=True)
# print(california_housing.DESCR)
# #overview of the entire dataset
# print(california_housing.frame.head())
# #features used by the training model without the y-variable (MedHouseVal)
# print(california_housing.data.head())
# #looking at the target to be predicted
# print(california_housing.target.head()) #target is the median of house value of each district- regression problem
# #to check other details and if the data contains any missing values
# # print(california_housing.frame.info())
# california_housing.frame.hist(figsize=(12,10),bins=30,edgecolor="black")
# plt.subplots_adjust(hspace=0.7,wspace=0.4)
# plt.show()
#focusing on the specific features
# features_of_interest=["AveRooms","AveBedrms","AveOccup","Population"]
# print(california_housing.frame[features_of_interest].describe())
import seaborn as sns
# sns.scatterplot(
#     data=california_housing.frame,
#     x="Longitude",
#     y="Latitude",
#     size="MedHouseVal",
#     hue="MedHouseVal",
#     palette="rocket",
#     alpha=0.5,
# )
# plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05,0.95), loc="upper left")
# _=plt.title(("Median house value depending of\n their spatial location"))
# plt.show()
#random subsampling- choosing 500 samples
import numpy as np
rng=np.random.RandomState(0)
indices=rng.choice(
    np.arange(california_housing.frame.shape[0]), size=100,replace=False
)
# sns.scatterplot(
#     data=california_housing.frame.iloc[indices],
#     x="Longitude",
#     y="Latitude",
#     size="MedHouseVal",
#     hue="MedHouseVal",
#     palette="viridis",
#     alpha=0.5,
# )
# plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 1), loc="upper left")
# _ = plt.title("Median house value depending of\n their spatial location")
# plt.show()
#making a pair plot- of all features and the target (only dropping longitude and latitude)
# import pandas as pd
# Drop the unwanted columns
# columns_drop = ["Longitude", "Latitude"]
# subset = california_housing.frame.iloc[indices].drop(columns=columns_drop)
# # Quantize the target and keep the midpoint for each interval
# subset["MedHouseVal"] = pd.qcut(subset["MedHouseVal"], 6, retbins=False)
# subset["MedHouseVal"] = subset["MedHouseVal"].apply(lambda x: x.mid)
# _ = sns.pairplot(data=subset, hue="MedHouseVal", palette="viridis")
# print(subset["MedHouseVal"])
# plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

alphas = np.logspace(-3, 1, num=30)
model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
cv_results = cross_validate(
    model,
    california_housing.data,
    california_housing.target,
    return_estimator=True,
    n_jobs=2,
)
print(cv_results)
score = cv_results["test_score"]
print(f"R2 score: {score.mean():.3f} ± {score.std():.3f}")
import pandas as pd

coefs = pd.DataFrame(
    [est[-1].coef_ for est in cv_results["estimator"]],
    columns=california_housing.feature_names,
)
color = {"whiskers": "black", "medians": "black", "caps": "black"}
coefs.plot.box(vert=False, color=color)
plt.axvline(x=0, ymin=-1, ymax=1, color="black", linestyle="--")
_ = plt.title("Coefficients of Ridge models\n via cross-validation")
plt.show()