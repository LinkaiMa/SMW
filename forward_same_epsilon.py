import numpy as np
import pandas as pd
from numpy.linalg import inv, norm
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def compute_SMW(n,k,A,B,U,V,lamda,e1_rel,e2_rel,alpha):
    A_inv = inv(A)
    A_inv_norm = norm(A_inv,ord=2)
    e1 = e1_rel*A_inv_norm
    noise1 = np.random.normal(size=(n,n))
    noise1 *= e1/norm(noise1,ord=2)
    A_til_inv = A_inv + noise1
    cap_til = inv(np.eye(k)+V.T@A_til_inv@U)
    e2 = e2_rel*norm(cap_til,ord=2)
    noise2 = np.random.normal(size=(k,k))
    noise2 *= e2/norm(noise2,ord=2)
    Z_inv = cap_til + noise2
    B_inv_approx = A_til_inv - A_til_inv@U@Z_inv@V.T@A_til_inv
    B_inv = inv(B) 
    error = norm(B_inv-B_inv_approx,ord=2)
    bound =  (4*e2_rel + 12*e1_rel)*A_inv_norm
    full_bound = e1 + e1*lamda*(2*A_inv_norm+e1)*alpha + lamda*(A_inv_norm+e1)**2 *(e2+2*e1*lamda*alpha**2)
    return error,bound,full_bound


def parse_args():
    parser = argparse.ArgumentParser(description='SMW Forward Error Experiment')
    parser.add_argument('--small_update', action='store_true', default=True,
                        help='Use small update regime (default: True)')
    parser.add_argument('--large_update', dest='small_update', action='store_false',
                        help='Use large update regime')
    parser.add_argument('--n', type=int, default=100, help='Matrix size n (default: 100)')
    parser.add_argument('--k', type=int, default=20, help='Rank k (default: 20)')
    parser.add_argument('--num_experiments', type=int, default=100, help='Number of experiments (default: 100)')
    parser.add_argument('--rand_seed', type=int, default=0, help='Random seed (default: 0)')
    return parser.parse_args()

args = parse_args()
small_update = args.small_update
n = args.n
k = args.k
num_experiments = args.num_experiments
rand_seed = args.rand_seed
lamda_scale = 0.4
scale = 1
es = np.logspace(-6,0,20) if small_update else np.logspace(-10,-4,20)
title_str = f'forward_error_n={n}_k={k}_lambda={lamda_scale}'
np.random.seed(rand_seed)
A = np.random.normal(size=(n,n),scale= scale)
U = np.random.normal(size=(n,k))
U /= norm(U, ord=2)
V = np.random.normal(size=(n,k))
V /= norm(V, ord=2)
svals = np.linalg.svd(A,compute_uv=False)
sigma_min, sigma_max = np.min(svals), np.max(svals)
lamda = lamda_scale*(sigma_min if small_update else sigma_max)
B = A + lamda*U@V.T
alpha = norm(inv(np.eye(k)+V.T@inv(A)@U*lamda),ord=2)

errors_samples      = np.zeros((len(es), num_experiments),dtype=float)
bounds_samples      = np.zeros_like(errors_samples,dtype=float)
full_bounds_samples = np.zeros_like(errors_samples,dtype=float)

for i,e in enumerate(es):
    for t in range(num_experiments):
        errors_samples[i,t],bounds_samples[i,t],full_bounds_samples[i,t]= compute_SMW(n,k,A,B,np.sqrt(lamda)*U,np.sqrt(lamda)*V,lamda,e,e,alpha)


df = pd.DataFrame({
    'epsilon': np.repeat(es, num_experiments),
    'error':   errors_samples.flatten(),
    'bound':   bounds_samples.flatten(),
    'full':    full_bounds_samples.flatten(),
})
df_long = df.melt(
    id_vars='epsilon',
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
    'full':  (4, 2, 1, 2)   # dash-dot
}
palette = {
    'error': 'C0',
    'bound': 'C1',
    'full':  'C2'
}
label_map = {
    'error': r"$\|\widetilde B^{-1} - B^{-1}\|_2$",
    'bound': r"$4\|A^{-1}\|_2\ (3\,\epsilon^{\text{rel}}_1 + \epsilon^{\text{rel}}_2)$",
    'full':  "forward error bound"
}

# ── Single Seaborn call ─────────────────────────────────────────────────────
sns.set_style("whitegrid")
fig, ax = plt.subplots() #figsize=(8,6)

sns.lineplot(
    data=df_long,
    x="epsilon", y="value",
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
    1/(2*lamda*alpha*norm(inv(A),ord=2)),
    linestyle="--", color="red",
    label=r"$\epsilon_1^{\text{abs}} = \frac{1}{2\lambda\,\alpha}$"
)

# ── Log scales, labels & title ───────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel("Value")
ax.set_title(f'Forward Error n = {n}, k = {k}, '+r'$\lambda$'+f' = {lamda_scale} '+(r'$\cdot\sigma_{\min}(A)$' if small_update else r'$\cdot\sigma_{\max}(A)$'))

# ── Rebuild legend with your LaTeX labels ────────────────────────────────────
handles, labels = ax.get_legend_handles_labels()
new_labels = [label_map.get(lab, lab) for lab in labels]
ax.legend(handles, new_labels, loc="best")
plt.tight_layout()
os.makedirs('Figure1', exist_ok=True)
plt.savefig('Figure1/'+'new_'+title_str+('_same_epsilon_small_update.pdf' if small_update else '_same_epsilon_large_update.pdf'))
plt.close()