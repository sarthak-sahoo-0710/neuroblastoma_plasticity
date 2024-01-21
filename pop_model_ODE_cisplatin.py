#%%
import random
import pandas as pd
import numpy as np

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the system of ODEs
def system_of_odes(t, y, k_a, g_a, k_aA, k_Aa, g_m, k_mM, k_Mm, k_am, k_ma, k_cisplatin_A, k_cisplatin_M):
    # Unpack the state variables
    a, A, m, M = y
    
    # Define the ODEs
    dadt = g_a*a*(1-((a+A)*(3/16000)+(m+M)*(1/254))/k_a) - k_aA*a + k_Aa*A - k_am*a + k_ma*m - k_cisplatin_A*a
    dAdt = g_a*A*(1-((a+A)*(3/16000)+(m+M)*(1/254))/k_a) + k_aA*a - k_Aa*A - k_cisplatin_A*A
    dmdt = g_m*m*(1-((a+A*(3/16000))+(m+M)*(1/254))/k_a) - k_mM*m + k_Mm*M + k_am*a - k_ma*m - k_cisplatin_M*m
    dMdt = g_m*M*(1-((a+A)*(3/16000)+(m+M)*(1/254))/k_a) + k_mM*m - k_Mm*M - k_cisplatin_M*M
    
    # Return the derivatives
    return [dadt, dAdt, dmdt, dMdt]

def simulation(params):
    # Set initial conditions
    initial_conditions = [31,31,19,19]  

    # Create a time span
    t_span = (0, 150)  # Time span from 0 to 10

    # Solve the ODEs using solve_ivp
    solution = solve_ivp(system_of_odes, t_span, initial_conditions, t_eval=np.linspace(0, 150, 100), args=params)

    # Extract the solutions for x and z
    t = solution.t
    a_solution = solution.y[0]*(3/16000)
    A_solution = solution.y[1]*(3/16000)
    m_solution = solution.y[2]*(1/254)
    M_solution = solution.y[3]*(1/254)
    """tot_sol = (solution.y[0]+solution.y[1]+solution.y[2]+solution.y[3])/k_a
    adrn_sol = (solution.y[0]+solution.y[1])/k_a
    mes_sol = (solution.y[2]+solution.y[3])/k_a
    plastic_sol = (solution.y[0]+solution.y[2])/k_a"""
    tot_sol = ((solution.y[0]+solution.y[1])*(3/16000)+(solution.y[2]+solution.y[3])*(1/254))
    adrn_sol = (solution.y[0]+solution.y[1]*(3/16000))
    mes_sol = (solution.y[2]+solution.y[3]*(1/254))
    plastic_sol = (solution.y[0]*(3/16000)+solution.y[2]*(1/254))

    return t, a_solution, A_solution, m_solution, M_solution, tot_sol, adrn_sol, mes_sol, plastic_sol
#%%
x = []
y = []
for i in range(500):

    if i%10 == 0:
        print(i)
    
    k_a = 100

    g_a = random.random()
    k_aA = random.random()
    k_Aa = random.random()

    g_m = g_a/2
    k_mM = random.random()
    k_Mm = random.random()

    k_am = random.random()
    k_ma = random.random()

    k_cisplatin_A = random.random()
    k_cisplatin_M = k_cisplatin_A/2

    params = [k_a, g_a, k_aA, k_Aa, g_m, k_mM, k_Mm, k_am, k_ma, k_cisplatin_A, k_cisplatin_M]

    t, a_solution, A_solution, m_solution, M_solution, tot_sol, adrn_sol, mes_sol, plastic_sol = simulation(params)
    data = params+[tot_sol[-1], adrn_sol[-1], mes_sol[-1], plastic_sol[-1]]
    x.append(data)

    params = [k_a, g_a, k_aA, k_Aa, g_m, k_mM, k_Mm, k_am, k_ma, 0, 0]
    t, a_solution, A_solution, m_solution, M_solution, tot_sol, adrn_sol, mes_sol, plastic_sol = simulation(params)
    data = params+[tot_sol[-1], adrn_sol[-1], mes_sol[-1], plastic_sol[-1]]
    y.append(data)

#%%

df = pd.DataFrame(x)
df.columns = ["KC","g_a", "k_aA", "k_Aa", "g_m", "k_mM", "k_Mm", "k_am", "k_ma", "k_cisplatin_A", "k_cisplatin_M", "tot", "adrn", "mes", "plastic"]
df.to_csv("x_con.tsv",sep="\t")

df2 = pd.DataFrame(y)
df2.columns = ["KC","g_a", "k_aA", "k_Aa", "g_m", "k_mM", "k_Mm", "k_am", "k_ma", "k_cisplatin_A", "k_cisplatin_M", "tot", "adrn", "mes", "plastic"]
df2.to_csv("y_con.tsv",sep="\t")

#%%
from sklearn.linear_model import LinearRegression

a = df2[["g_a", "k_aA", "k_Aa", "g_m", "k_mM", "k_Mm", "k_am", "k_ma","k_cisplatin_A", "k_cisplatin_M"]]
b = df2["mes"]

model = LinearRegression()
model.fit(a, b)

r2_score = model.score(a, b)
print(f"R-squared value: {r2_score}")

for i in list(model.coef_):
    print(i)

#%%

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(t, a_solution, label='a(t)')
plt.plot(t, A_solution, label='A(t)')
plt.plot(t, m_solution, label='m(t)')
plt.plot(t, M_solution, label='M(t)')

plt.xlabel('Time')
plt.ylabel('Solution')
plt.legend()
plt.title('Solution of Coupled ODEs using solve_ivp')
plt.grid(True)
plt.show()
plt.close()

plt.plot(t, adrn_sol, label='adrn(t)', color='purple')
plt.plot(t, mes_sol, label='mes(t)', color='orange')
plt.plot(t, tot_sol, label='tot(t)', color='black')
plt.xlabel('Time')
plt.ylabel('Solution')
plt.legend()
plt.title('Solution of Coupled ODEs using solve_ivp')
plt.grid(True)
plt.show()
plt.close()


#%%
