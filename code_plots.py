#%%
import random
import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from matplotlib.pyplot import rcParams

rcParams.update({'font.family':'Arial', 'font.size':16})
# %%
df_no_treatment = pd.read_csv("v2_pop_model_ODE_y_gmby2_confluency5k_half.tsv",sep="\t",index_col=0)
df_no_treatment.drop(columns=["k_cisplatin_A", "k_cisplatin_M"], inplace=True)
df_treatment = pd.read_csv("v2_pop_model_ODE_x_gmby2_confluency5k_half.tsv",sep="\t",index_col=0)
df = pd.concat([df_no_treatment,df_treatment[["k_cisplatin_A","k_cisplatin_M","tot","adrn","mes","plastic"]]],axis=1)
df.columns = list(df.columns[:-4]) + ["tot_treated","adrn_treated","mes_treated","plastic_treated"]

df["difference"] = df["k_Mm"]+df["k_ma"]+df["k_aA"] - (df["k_Aa"] + df["k_am"] + df["k_mM"])
# %%
sns.scatterplot(x=df["tot_treated"],y=df["mes_treated"], linewidth=0, s = 10)
#%%
from sklearn.linear_model import LinearRegression

variables = ["g_a", "g_m", "k_am", "k_ma", "k_aA", "k_Aa", "k_mM", "k_Mm", "k_cisplatin_A", "k_cisplatin_M"] 
a = df_case[variables]
b = df_case["mes_treated"] - df_case["mes"] 

model = LinearRegression()
model.fit(a, b)

r2_score = model.score(a, b)
print(f"R-squared value: {r2_score}")
print(f"intercept: {model.intercept_}")

for idx, val in enumerate(list(model.coef_)):
    print(variables[idx],"\t",val)

y_pred = model.predict(a)
sns.scatterplot(x=b, y=y_pred, color='black', linewidth=0, s = 10)
sns.regplot(x=b, y=y_pred, color='red', scatter=False)

# %%
sns.kdeplot(df["tot"],fill=True, color='black')
sns.kdeplot(df["tot_treated"],fill=True, color='red')
# %%
tot_instant = df["mes_treated"]+df["adrn_treated"]
sns.scatterplot(x=np.log10(df["mes_treated"]/tot_instant), y=df["plastic_treated"], color='black', linewidth=0, s = 10)
sns.regplot(x=np.log10(df["mes_treated"]/tot_instant), y=df["plastic_treated"], color='red', scatter=False)
# %%
sns.scatterplot(x=df["mes"], y=df["mes_treated"], color='black', linewidth=0, s = 10)
sns.lineplot(x=[0,1],y=[0,1],color="red")
# %%
df_case = df[df["mes_treated"] > df["mes"]]
df_control = df[df["mes_treated"] < df["mes"]]

for i in list(df_case.columns):
    x,y = ss.ttest_ind(df_case[i], df_control[i])
    if y < 1e-5:
        print(i)
# %%
parameter = "g_m"
sns.kdeplot(df_control[parameter],fill=True, color='black')
sns.kdeplot(df_case[parameter],fill=True, color='red')
# %%
parameter = "k_cisplatin_M"
sns.kdeplot(df_control["k_Aa"]-df_control["k_aA"],fill=True, color='black')
sns.kdeplot(df_case["k_Aa"]-df_case["k_aA"],fill=True, color='red')
# %%
sns.kdeplot(df_control["k_Mm"]-df_control["k_mM"],fill=True, color='black')
sns.kdeplot(df_case["k_Mm"]-df_case["k_mM"],fill=True, color='red')
# %%
sns.kdeplot(df_control["mes_treated"],fill=True, color='black')
sns.kdeplot(df_case["mes_treated"],fill=True, color='red')
# %%
sns.kdeplot(df_control["adrn_treated"],fill=True, color='black')
sns.kdeplot(df_case["adrn_treated"],fill=True, color='red')
# %%
sns.kdeplot(df_control["plastic_treated"],fill=True, color='black')
sns.kdeplot(df_case["plastic_treated"],fill=True, color='red')
# %%
