import numpy as np
import pandas as pd
from numpy.linalg import inv, norm, qr
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse


def compute_SMW(n,k,A,A_inv,B,U,V,lamda,e1,e2):
    A_inv_norm = norm(A_inv,ord=2)
    noise1 = np.random.normal(size=(n,n))
    noise1 *= e1/norm(noise1,ord=2)
    A_til_inv = A_inv + noise1
    cap_til = inv(np.eye(k)+V.T@A_til_inv@U)
    alpha = norm(inv(np.eye(k)+V.T@A_inv@U),ord=2)
    noise2 = np.random.normal(size=(k,k))
    noise2 *= e2/norm(noise2,ord=2)
    Z_inv = cap_til + noise2
    B_inv_approx = A_til_inv - A_til_inv@U@Z_inv@V.T@A_til_inv
    B_inv = inv(B) 
    error = norm(B_inv-B_inv_approx,ord=2)
    bound =  2*lamda**2 *e1*A_inv_norm**2*alpha**2 
    full_bound = e1 + e1*lamda*(2*A_inv_norm+e1)*alpha + lamda*(A_inv_norm+e1)**2 *(e2+2*e1*lamda*alpha**2)
    return error,bound,full_bound

def parse_args():
    parser = argparse.ArgumentParser(description='SMW Forward Error Experiment for controlled alpha')
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
epsilon = 1e-3 if small_update else 1e-10
S_A = np.logspace(2, -2, n)
sigma_max, sigma_min = np.max(S_A), np.min(S_A)
lamda = 2*sigma_min if small_update else 2 * sigma_max
index_list = list(range(int(n / 4), int(n * 3 / 4), 20))

alphas = np.logspace(1,6,len(index_list)) if small_update else np.logspace(1,9,len(index_list))
errors_samples      = np.zeros((len(alphas), num_experiments),dtype=float)
bounds_samples      = np.zeros_like(errors_samples,dtype=float)
full_bounds_samples = np.zeros_like(errors_samples,dtype=float)
rand_seed = args.rand_seed
np.random.seed(rand_seed)
U_A,_ = qr(np.random.randn(n,n))
V_A,_ = qr(np.random.randn(n,n))
A = U_A @ np.diag(S_A) @ V_A.T
A_inv = inv(A)
I = np.eye(n)
Q = I[:,-k:]
for i, index in enumerate(index_list):
    alpha = alphas[i]
    error_total = 0.0
    bound_total = 0.0
    full_bound_total = 0.0
    U = U_A@Q
    S_diag = np.zeros(n)
    S_diag[-k] = lamda
    for j in range(n-k+1,n-1):
        S_diag[j] = abs(1/alpha-1)*S_A[j]*1.001 #(1/alpha-1)*S_A[j] #
    S_diag[-1] = (1/alpha-1)*S_A[-1]
    S = np.diag(S_diag)
    V = V_A@S@Q
    B = A + U @ V.T
    assert abs(alpha-norm(inv(np.eye(k)+V.T@inv(A)@U),ord=2))/alpha< 1e-3
    assert abs(lamda-norm(U,ord=2)*norm(V,ord=2))/lamda<1e-3
    for t in range(num_experiments):
        errors_samples[i,t],bounds_samples[i,t],full_bounds_samples[i,t]= compute_SMW(n, k, A, A_inv, B, U, V, lamda, epsilon, epsilon)

df = pd.DataFrame({
    'alpha': np.repeat(alphas, num_experiments),
    'error':   errors_samples.flatten(),
    'bound':   bounds_samples.flatten(),
    'full':    full_bounds_samples.flatten(),
})
df_long = df.melt(
    id_vars='alpha',
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
    'full':  (4, 2, 1, 2)    # dash-dot
}
palette = {
    'error': 'C0',
    'bound': 'C1',
    'full':  'C2'
}
label_map = {
    'error': r"$\|\widetilde B^{-1} - B^{-1}\|_2$",
    'bound': r"$2\|A^{-1 }\|_2^2 \lambda^2 \alpha^2 \epsilon^{\text{abs}}_1$",
    'full':  "forward error bound"
}

# ── Single Seaborn call ─────────────────────────────────────────────────────
sns.set_style("whitegrid")
fig, ax = plt.subplots() #figsize=(8,6)

sns.lineplot(
    data=df_long,
    x="alpha", y="value",
    hue="metric", style="metric",
    markers=marker_styles,
    dashes=line_styles,
    palette=palette,
    estimator="mean",
    errorbar=("ci", 95),      # 95% bootstrap CI
    ax=ax
)

# ── Add the red vertical threshold line ─────────────────────────────────────
ax.axvline(
    1/(2*lamda*epsilon),
    linestyle="--", color="red",
    label=r"$\alpha = \frac{1}{2 \lambda \epsilon^{\text{abs}}_1}$"
)
# ── Log scales, labels & title ───────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Value")
ax.set_title("Forward Error "+(r'$\lambda = 2\sigma_{\min}(A), \epsilon^{\text{abs}}_1 = \epsilon^{\text{abs}}_2 = $'if small_update else r'$\lambda = 2\sigma_{\max}(A), \epsilon^{\text{abs}}_1 = \epsilon^{\text{abs}}_2 = $')+str(epsilon))

# ── Rebuild legend with your LaTeX labels ────────────────────────────────────
handles, labels = ax.get_legend_handles_labels()
new_labels = [label_map.get(lab, lab) for lab in labels]
ax.legend(handles, new_labels, loc="best")
plt.tight_layout()
os.makedirs('Figure4', exist_ok=True)
plt.savefig("Figure4/old_diagonal_forward_error_small_update.pdf" if small_update else "Figure4/old_diagonal_forward_error_large_update.pdf")
plt.close()

