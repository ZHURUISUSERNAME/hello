import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Initialize decision variables
P_CHP_e = cp.Variable((1, 24))
P_CHP_h = cp.Variable((1, 24))
P_g_CHP = cp.Variable((1, 24))
P_e_EL = cp.Variable((1, 24))
P_EL_H = cp.Variable((1, 24))
P_H_MR = cp.Variable((1, 24))
P_MR_g = cp.Variable((1, 24))
P_H_HFC = cp.Variable((1, 24))
P_HFC_e = cp.Variable((1, 24))
P_HFC_h = cp.Variable((1, 24))
P_DG = cp.Variable((1, 24))
P_g_GB = cp.Variable((1, 24))
P_GB_h = cp.Variable((1, 24))

# Storage variables
P_ES1_cha, P_ES2_cha, P_ES3_cha, P_ES4_cha = [cp.Variable((1, 24)) for _ in range(4)]
P_ES1_dis, P_ES2_dis, P_ES3_dis, P_ES4_dis = [cp.Variable((1, 24)) for _ in range(4)]
S_1, S_2, S_3, S_4 = [cp.Variable((1, 24)) for _ in range(4)]
B_ES1_cha, B_ES2_cha, B_ES3_cha, B_ES4_cha = [cp.Variable((1, 24), boolean=True) for _ in range(4)]
B_ES1_dis, B_ES2_dis, B_ES3_dis, B_ES4_dis = [cp.Variable((1, 24), boolean=True) for _ in range(4)]

P_e_buy = cp.Variable((1, 24))
P_g_buy = cp.Variable((1, 24))

# Import constant parameters
P_e_load = 1.1 * np.array([717.451523545706, 695.290858725762, 689.750692520776, 698.060941828255, 745.152354570637, 808.864265927978, 836.565096952909, 872.576177285319, 886.426592797784, 900.277008310249, 894.736842105263, 883.656509695291, 875.346260387812, 864.265927977839, 864.265927977839, 868.421052631579, 876.731301939058, 889.196675900277, 880.886426592798, 864.265927977839, 836.565096952909, 817.174515235457, 772.853185595568, 745.152354570637])
P_h_load = 0.9 * np.array([864.265927977839, 941.828254847645, 958.448753462604, 955.678670360111, 988.919667590028, 997.229916897507, 903.047091412742, 833.795013850416, 786.703601108033, 703.601108033241, 664.819944598338, 626.038781163435, 595.567867036011, 590.027700831025, 565.096952908587, 639.889196675900, 714.681440443213, 806.094182825485, 811.634349030471, 831.024930747922, 811.634349030471, 808.864265927978, 800.554016620499, 808.864265927978])
P_g_load = np.array([229.916897506925, 224.376731301939, 216.066481994460, 221.606648199446, 224.376731301939, 252.077562326870, 268.698060941828, 288.088642659280, 299.168975069252, 288.088642659280, 293.628808864266, 282.548476454294, 279.778393351801, 271.468144044321, 271.468144044321, 268.698060941828, 277.008310249307, 293.628808864266, 307.479224376731, 304.709141274238, 293.628808864266, 285.318559556787, 277.008310249307, 265.927977839335])
P_DG_max = 1.2 * np.array([850.415512465374, 864.265927977839, 886.426592797784, 891.966759002770, 894.736842105263, 849.030470914127, 833.795013850416, 653.739612188366, 556.786703601108, 501.385041551247, 432.132963988920, 310.249307479224, 240.997229916897, 252.077562326870, 265.927977839335, 296.398891966759, 343.490304709141, 354.570637119114, 426.592797783934, 526.315789473684, 675.900277008310, 742.382271468144, 854.570637119114, 878.116343490305]) + 275 * np.random.rand(24)
c_e_buy = 1.2 * np.array([0.38]*7 + [0.68]*4 + [1.2]*3 + [0.68]*4 + [1.2]*4 + [0.38]*2)
c_g_buy = 0.45 * np.ones(24)
P_DG_max = P_DG_max.reshape(1, -1)

# Define constraints
constraints = [
    P_CHP_e == 0.92 * P_g_CHP,
    0 <= P_g_CHP, P_g_CHP <= 600,
    0.5 * P_CHP_e <= P_CHP_h, P_CHP_h <= 2.1 * P_CHP_e,
    -0.2 * 600 <= P_g_CHP[:, 1:] - P_g_CHP[:, :-1], P_g_CHP[:, 1:] - P_g_CHP[:, :-1] <= 0.2 * 600,
    P_EL_H == 0.88 * P_e_EL,
    0 <= P_e_EL, P_e_EL <= 500,
    -0.2 * 500 <= P_e_EL[:, 1:] - P_e_EL[:, :-1], P_e_EL[:, 1:] - P_e_EL[:, :-1] <= 0.2 * 500,
    P_MR_g == 0.6 * P_H_MR,
    0 <= P_H_MR, P_H_MR <= 250,
    -0.2 * 250 <= P_H_MR[:, 1:] - P_H_MR[:, :-1], P_H_MR[:, 1:] - P_H_MR[:, :-1] <= 0.2 * 250,
    P_HFC_e == 0.85 * P_H_HFC,
    0 <= P_H_HFC, P_H_HFC <= 250,
    0.5 * P_HFC_e <= P_HFC_h, P_HFC_h <= 2.1 * P_HFC_e,
    -0.2 * 250 <= P_H_HFC[:, 1:] - P_H_HFC[:, :-1], P_H_HFC[:, 1:] - P_H_HFC[:, :-1] <= 0.2 * 250,
    0 <= P_DG, P_DG <= P_DG_max,  # This line should now work correctly
    P_GB_h == 0.95 * P_g_GB,
    0 <= P_g_GB, P_g_GB <= 800,
    -0.2 * 800 <= P_g_GB[:, 1:] - P_g_GB[:, :-1], P_g_GB[:, 1:] - P_g_GB[:, :-1] <= 0.2 * 800,
]

# Storage constraints
for ES, P_cha, P_dis, B_cha, B_dis, S, cap in [
    (1, P_ES1_cha, P_ES1_dis, B_ES1_cha, B_ES1_dis, S_1, 450),
    (2, P_ES2_cha, P_ES2_dis, B_ES2_cha, B_ES2_dis, S_2, 500),
    (3, P_ES3_cha, P_ES3_dis, B_ES3_cha, B_ES3_dis, S_3, 150),
    (4, P_ES4_cha, P_ES4_dis, B_ES4_cha, B_ES4_dis, S_4, 200),
]:
    constraints.extend([
        0 <= P_cha, P_cha <= B_cha * 0.5 * cap,
        0 <= P_dis, P_dis <= B_dis * 0.5 * cap,
        S[:, 0] == 0.3 * cap,
        S[:, -1] == S[:, 0],
        B_cha + B_dis <= 1,
        0.2 * cap <= S, S <= 0.9 * cap,
        S[:, 1:] == S[:, :-1] + 0.95 * P_cha[:, 1:] - P_dis[:, 1:] / 0.95,
    ])

# Reshape P_e_load, P_h_load, and P_g_load to match the shape of other variables
P_e_load = P_e_load.reshape(1, -1)
P_h_load = P_h_load.reshape(1, -1)
P_g_load = P_g_load.reshape(1, -1)

# Power balance constraints
constraints.extend([
    P_e_buy == P_e_load + P_e_EL + P_ES1_cha - P_ES1_dis - P_DG - P_CHP_e - P_HFC_e,
    P_HFC_h + P_CHP_h + P_GB_h == P_h_load + P_ES2_cha - P_ES2_dis,
    P_g_buy == P_g_load + P_ES3_cha - P_ES3_dis + P_g_CHP + P_g_GB - P_MR_g,
    P_EL_H == P_H_MR + P_H_HFC + P_ES4_cha - P_ES4_dis,
    0 <= P_e_buy, P_e_buy <= 5000,
    0 <= P_g_buy, P_g_buy <= 5000,
])

# Ensure c_e_buy and c_g_buy are reshaped to (1, 24)
c_e_buy = c_e_buy.reshape(1, -1)
c_g_buy = c_g_buy.reshape(1, -1)

# Objective function
E_e_buy = 0.68 * cp.sum(P_e_buy)
E_CHP = 0.102 * 3.6 * cp.sum(P_CHP_h + 6/3.6 * P_CHP_e)
E_GB = 0.102 * 3.6 * cp.sum(P_GB_h)
E_IES = E_e_buy + E_CHP + E_GB

E_e_buy_a = 1.08 * cp.sum(P_e_buy)
E_CHP_a = 0.065 * 3.6 * cp.sum(P_CHP_h + 6/3.6 * P_CHP_e)
E_GB_a = 0.065 * 3.6 * cp.sum(P_GB_h)
E_MR_a = 1 * cp.sum(P_MR_g)
E_IES_a = E_e_buy_a + E_CHP_a + E_GB_a
E = E_IES_a - E_MR_a

E_v = cp.Variable(5)
lamda = 0.250
constraints.extend([
    E == cp.sum(E_v),
    0 <= E_v[:4], E_v[:4] <= 2000,
    0 <= E_v[4],
])

C_CO2 = cp.sum([(lamda + i * 0.25 * lamda) * E_v[i] for i in range(5)])

# Use cp.multiply for element-wise multiplication
Cost = cp.sum(cp.multiply(c_e_buy, P_e_buy) + cp.multiply(c_g_buy, P_g_buy)) + C_CO2 + 0.13 * cp.sum(P_DG_max - P_DG)

# Solve the problem
problem = cp.Problem(cp.Minimize(Cost), constraints)
problem.solve(solver=cp.SCIP, verbose=True)
print(f"Optimal cost: {problem.value}")

# Plotting results (example for electric power balance)
Plot_e = np.vstack([P_e_buy.value, P_HFC_e.value, P_ES1_dis.value, P_DG.value, P_CHP_e.value, P_e_EL.value, P_ES1_cha.value])
labels = ['Buy', 'HFC', 'Discharge', 'Wind', 'CHP', 'EL', 'Charge']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

plt.figure(figsize=(12, 6))

# Create the stacked bar chart
bottom = np.zeros(24)
for i, row in enumerate(Plot_e):
    plt.bar(range(24), row[0], bottom=bottom, label=labels[i], color=colors[i])
    bottom += row[0]

plt.plot(range(24), P_e_load[0], 'r-o', linewidth=1.5, label='Load')
plt.xlabel('Time (h)')
plt.ylabel('Power (kW)')
plt.title('Electric Power Balance')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()