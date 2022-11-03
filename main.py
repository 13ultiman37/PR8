import pandas as pd
from pandas import DataFrame
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

print("\n----------------------------1 часть--------------------------------\n")
data = pd.read_csv("insurance.csv")
print("Количество пустых значений:")
print(data.isna().sum())
data = data.drop_duplicates()
print("\nКоличество дубликатов: ", data.duplicated().sum())
print("\nСписок уникальных регионов: ", data.region.unique())
print("\n", data.describe())

print("\n----------------------------2 часть--------------------------------\n")

citizen_region = data["region"]
citizen_bmi = data["bmi"]

df = pd.DataFrame({"region": citizen_region, "bmi": citizen_bmi})
groups = df.groupby("region").groups
southwest = citizen_bmi[groups["southwest"]]
southeast = citizen_bmi[groups["southeast"]]
northwest = citizen_bmi[groups["northwest"]]
northeast = citizen_bmi[groups["northeast"]]


print("Первый способ, scipy: ", stats.f_oneway(southwest, southeast, northwest, northeast))

print("\n----------------------------3 часть--------------------------------\n")
model = ols('bmi ~ region', data=df).fit()
anova_result = sm.stats.anova_lm(model, typ=2)
print("\nВторой способ, anova_lm: ")
print(anova_result)

print("\n----------------------------4 часть--------------------------------\n")
regions = ["southwest", "southeast", "northwest", "northeast"]
region_pairs = []
for region1 in range(3):
    for region2 in range(region1 + 1, 4):
        region_pairs.append((regions[region1], regions[region2]))

for region1, region2 in region_pairs:
    print(region1, region2)
    print(stats.ttest_ind(df.bmi[groups[region1]], df.bmi[groups[region2]]))

print("\n----------------------------5 часть--------------------------------\n")
tukey = pairwise_tukeyhsd(endog=df.bmi, groups=df.region, alpha=0.05)
tukey.plot_simultaneous()
#plt.vlines(x=49.57, ymin=-0.5, ymax=4.5, color="blue")
print(tukey.summary())
plt.show()

print("\n----------------------------6 часть--------------------------------\n")
dt = pd.DataFrame({"sex": data["sex"], "region": data["region"], "bmi": data["bmi"]})
model2 = ols('bmi ~ C(region) + C(sex) + C(region):C(sex)', data=dt).fit()
print(sm.stats.anova_lm(model2, typ=2))

print("\n----------------------------7 часть--------------------------------\n")
dt['combination'] = dt.sex + " / " + dt.region
tukey2 = pairwise_tukeyhsd(endog=dt['bmi'], groups=dt['combination'], alpha=0.05)
tukey2.plot_simultaneous()
#plt.vlines(x=49.57, ymin=-0.5, ymax=4.5, color="blue")
print(tukey2.summary())
plt.show()
