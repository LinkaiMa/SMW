import numpy as np
import pandas as pd
from numpy.linalg import inv, norm, qr
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def compute_SMW(n,k,A,B,U,V,lamda,e1,e2):
    A_norm = norm(A,ord=2)
    A_inv = inv(A)
    A_inv_norm = norm(A_inv,ord=2)
    noise1 = np.random.normal(size=(n,n))
    noise1 *= e1/norm(noise1,ord=2)
    A_til_inv = A_inv + noise1
    cap_til = inv(np.eye(k)+V.T@A_til_inv@U)
    noise2 = np.random.normal(size=(k,k))
    noise2 *= e2/norm(noise2,ord=2)
    Z_inv = cap_til + noise2
    B_inv_approx = A_til_inv - A_til_inv@U@Z_inv@V.T@A_til_inv
    beta = np.linalg.norm(np.eye(k)+V.T@A_inv@U,ord=2)
    error = norm(B-inv(B_inv_approx),ord=2)
    bound = 4*lamda*e2*beta**2
    full_bound = 2*e1*A_norm**2 + 4*lamda*e2*(beta+lamda*e1)**2
    return error,bound,full_bound

def parse_args():
    parser = argparse.ArgumentParser(description='SMW Backward Error Experiment for controlled beta')
    parser.add_argument('--small_update', action='store_true', default=True,
                        help='Use small update regime (default: True)')
    parser.add_argument('--large_update', dest='small_update', action='store_false',
                        help='Use large update regime')
    parser.add_argument('--n', type=int, default=1000, help='Matrix size n (default: 100)')
    parser.add_argument('--k', type=int, default=20, help='Rank k (default: 20)')
    parser.add_argument('--num_experiments', type=int, default=100, help='Number of experiments (default: 100)')
    parser.add_argument('--rand_seed', type=int, default=0, help='Random seed (default: 0)')
    return parser.parse_args()


args = parse_args()
small_update = args.small_update
n = args.n
k = args.k
num_experiments = args.num_experiments
epsilon = 1e-6
S_A = list(np.logspace(-2,-8,int(n*0.6)))+[1e-8 for i in range(n-int(n*0.6))] if small_update else np.logspace(2,-2,n)
A = np.diag(S_A)
sigma_max, sigma_min = np.max(S_A), np.min(S_A)
lamda = 100*sigma_min if small_update else 2*sigma_max 
assert epsilon < 1/2/sigma_max
index_list = list(range(int(n/4),int(n*3/4),20))
betas = np.zeros((len(index_list),num_experiments),dtype=float)
errors_samples      = np.zeros_like(betas,dtype=float)
bounds_samples      = np.zeros_like(betas,dtype=float)
full_bounds_samples = np.zeros_like(betas,dtype=float)
rand_seed = args.rand_seed
np.random.seed(rand_seed)
cond3_idx = 0
for i, index in enumerate(index_list):
    I = np.eye(n)
    Q = I[:, index:index + 20]
    U = A @ Q
    U_norm = norm(U, ord=2)
    V_norm = lamda / U_norm
    cond3_fail = False
    for t in range(num_experiments):
        # Generate a new S for each experiment
        S = np.random.randn(k, k)
        S *= V_norm / norm(S, ord=2)
        V = Q @ S.T
        # Compute beta for this experiment
        beta = norm(np.eye(k) + V.T @ inv(A) @ U, ord=2)
        betas[i,t] = beta
        B = A + U @ V.T
        err, b, fb = compute_SMW(n, k, A, B, U, V, lamda, epsilon, epsilon)
        assert epsilon < 1/2/(beta+lamda*epsilon)
        if 2*(beta+lamda*epsilon)**2 * epsilon >=0.5:
            cond3_fail = True
        errors_samples[i,t] = err
        bounds_samples[i,t] = b
        full_bounds_samples[i,t] = fb
    if not cond3_fail:
        cond3_idx = i

df = pd.DataFrame({
    'beta':   np.repeat(np.mean(betas,axis=1),num_experiments),
    'error':   errors_samples.flatten(),
    'bound':   bounds_samples.flatten(),
    'full':    full_bounds_samples.flatten(),
})
df_long = df.melt(
    id_vars='beta',
    value_vars=['error','bound','full'],
    var_name='metric',
    value_name='value'
)

# ── Style mappings ───────────────────────────────────────────────────────────
marker_styles = {
    'error': 'o',   # blue circle
    'bound': 's',   # orange square
    'full':  '^'    # green triangle
}
# Use () for solid, (1,1) for dotted, (4, 2, 1, 2) for dash-dot
line_styles = {
    'error': (),      # solid
    'bound': (1,1),   # dotted
    'full':  (4, 2, 1, 2)  # dash-dot
}
palette = {
    'error': 'C0',
    'bound': 'C1',
    'full':  'C2'
}
label_map = {
    'error': r"$\|B-\widetilde{B}\|_2$",
    'bound': r"$ 4\lambda\epsilon^{\text{abs}}_2 \beta^2$",
    'full':  "backward error bound"
}

# ── Single Seaborn call ─────────────────────────────────────────────────────
sns.set_style("whitegrid")
fig, ax = plt.subplots() #figsize=(8,6)

sns.lineplot(
    data=df_long,
    x="beta", y="value",
    hue="metric", style="metric",
    markers=marker_styles,
    dashes=line_styles,
    palette=palette,
    estimator="mean",
    errorbar=("ci", 95),      # 95% bootstrap CI
    ax=ax
)

# # ── Add the red vertical threshold line ─────────────────────────────────────
ax.axvline(
    np.mean(betas[cond3_idx,:]),
    linestyle="--", color="red",
    label=r'$2(\beta+\lambda \epsilon^{\text{abs}}_1)^2\epsilon^{\text{abs}}_2 \leq \frac{1}{2}$'
)# ── Log scales, labels & title ───────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("Value")
ax.set_title("Backward Error "+r'$\lambda = 100\,\sigma_{\min}(A), \epsilon^{\text{abs}}_1 = \epsilon^{\text{abs}}_2 = $'+str(epsilon) if small_update else r'$\lambda = 2\,\sigma_{\max}(A), \epsilon^{\text{abs}}_1 = \epsilon^{\text{abs}}_2 = $'+str(epsilon))
# ── Rebuild legend with your LaTeX labels ────────────────────────────────────
handles, labels = ax.get_legend_handles_labels()
new_labels = [label_map.get(lab, lab) for lab in labels]
ax.legend(handles, new_labels, loc="best")
plt.tight_layout()
os.makedirs('Figure5', exist_ok=True)
plt.savefig("Figure5/old_diagonal_backward_error_small_update.pdf" if small_update else "Figure5/old_diagonal_backward_error_large_update.pdf")
plt.close()

