"""
PI-DeepONet (parametric) for 2D Poisson with Gaussian source on [0,1]^2:
    -Δu = f(x,y; x0, y0, nu),   u = 0 on boundary.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


# ============================================================
# Device + reproducibility
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Standardized directory for saving fields for later comparison plots
DATA_DIR = "outputs_pideeponet"
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = "pideeponet_poisson_model.pt"

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("Random seed set to:", SEED)


# ============================================================
# Problem definition
# ============================================================
def gaussian_source(x, y, x0, y0, nu):
    """
    Gaussian forcing:
      f(x,y) = (1/(2πν^2)) exp(-((x-x0)^2 + (y-y0)^2) / (2ν^2))

    x, y: (N,) tensors
    x0, y0, nu: scalars (tensor or float)
    returns: (N,)
    """
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return (1.0 / (2.0 * torch.pi * nu**2)) * torch.exp(-r2 / (2.0 * nu**2))


# Training parameter ranges (same across your three methods)
x0_min, x0_max = 0.4, 0.6
y0_min, y0_max = 0.4, 0.6
nu_min, nu_max = 0.05, 0.1


def sample_pde_param(nu_min_curr=None, nu_max_curr=None):
    """
    Sample p=(x0,y0,nu)
      x0,y0 uniform in [0.4,0.6]
      nu log-uniform in [nu_min_curr, nu_max_curr]
    """
    if nu_min_curr is None:
        nu_min_curr = nu_min
    if nu_max_curr is None:
        nu_max_curr = nu_max

    x0 = x0_min + (x0_max - x0_min) * torch.rand(1, device=device)
    y0 = y0_min + (y0_max - y0_min) * torch.rand(1, device=device)

    # log-uniform nu
    u = torch.rand(1, device=device)
    nu_min_t = torch.tensor(nu_min_curr, device=device)
    nu_max_t = torch.tensor(nu_max_curr, device=device)
    nu = 10.0 ** (
        torch.log10(nu_min_t)
        + (torch.log10(nu_max_t) - torch.log10(nu_min_t)) * u
    )

    p = torch.stack([x0, y0, nu], dim=-1)  # (1,3)
    return p


def sample_collocation_points_p_dependent(
    N_int, N_bc, p, alpha=None, sigma_factor=3.0, device=device
):
    """
    p-dependent interior sampling:
      - a fraction alpha near (x0,y0) with spread ~ sigma_factor*nu
      - rest uniform on [0,1]^2

    boundary sampling:
      - uniform points on each side (monitoring only; BC is hard-enforced)

    Returns:
      xy_int: (N_int,2)
      xy_bc:  (N_bc,2)
    """
    x0, y0, nu = p[0]

    if alpha is None:
        alpha = 0.9 if nu.item() < 0.06 else 0.7

    N_loc = int(alpha * N_int)
    N_uni = N_int - N_loc

    # Localized interior points
    if N_loc > 0:
        loc_x = x0 + sigma_factor * nu * torch.randn(N_loc, 1, device=device)
        loc_y = y0 + sigma_factor * nu * torch.randn(N_loc, 1, device=device)
        loc_x = loc_x.clamp(0.0, 1.0)
        loc_y = loc_y.clamp(0.0, 1.0)
        xy_loc = torch.cat([loc_x, loc_y], dim=1)
    else:
        xy_loc = torch.empty(0, 2, device=device)

    # Uniform interior points
    if N_uni > 0:
        uni_x = torch.rand(N_uni, 1, device=device)
        uni_y = torch.rand(N_uni, 1, device=device)
        xy_uni = torch.cat([uni_x, uni_y], dim=1)
    else:
        xy_uni = torch.empty(0, 2, device=device)

    xy_int = torch.cat([xy_loc, xy_uni], dim=0)

    # Boundary points
    if N_bc > 0:
        N_side = N_bc // 4
        t = torch.rand(N_side, 1, device=device)

        xb = torch.cat(
            [
                torch.zeros(N_side, 1, device=device),
                torch.ones(N_side, 1, device=device),
                t,
                t,
            ],
            dim=0,
        )
        yb = torch.cat(
            [
                t,
                t,
                torch.zeros(N_side, 1, device=device),
                torch.ones(N_side, 1, device=device),
            ],
            dim=0,
        )
        xy_bc = torch.cat([xb, yb], dim=1)
    else:
        xy_bc = torch.empty(0, 2, device=device)

    return xy_int, xy_bc


# ============================================================
# FDM "exact" solver (same as your other scripts)
# ============================================================
def poisson_exact_fdm(N=60, x0=0.5, y0=0.5, nu=0.05):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    f = (1.0 / (2.0 * np.pi * nu**2)) * np.exp(-r2 / (2.0 * nu**2))

    Nint = N - 2
    f_inner = f[1:-1, 1:-1].reshape(-1)

    e = np.ones(Nint)
    D2 = (np.diag(-2 * e) + np.diag(e[:-1], 1) + np.diag(e[:-1], -1)) / dx**2
    I = np.eye(Nint)

    L = np.kron(I, D2) + np.kron(D2, I)
    rhs = -f_inner

    u_inner = np.linalg.solve(L, rhs)
    u = np.zeros((N, N))
    u[1:-1, 1:-1] = u_inner.reshape(Nint, Nint)
    return x, y, u


# ============================================================
# Test cases (same as your KAPI / HyperPINN scripts)
# ============================================================
test_cases = [
    ("center_in_range",      0.50, 0.50, 0.07),
    ("off_center_in_range",  0.45, 0.55, 0.09),
    ("shifted_out_of_range", 0.30, 0.30, 0.06),
    ("narrow_out_of_range",  0.50, 0.50, 0.03),
]


# ============================================================
# DeepONet model definition
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=3, act=nn.Tanh):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), act()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ParametricDeepONet(nn.Module):
    """
    Branch input: p=(x0,y0,nu)   shape (1,3)
    Trunk input:  xy            shape (N,2)

    u_raw(xy;p) = sum_j trunk_j(xy) * branch_j(p) + bias
    Hard Dirichlet BC via:
      u(x,y;p) = x(1-x)y(1-y) * u_raw(x,y;p)
    """

    def __init__(self, latent_dim=128, hidden_dim=128, depth=3):
        super().__init__()
        self.branch = MLP(3, hidden_dim, latent_dim, depth=depth, act=nn.Tanh)
        self.trunk  = MLP(2, hidden_dim, latent_dim, depth=depth, act=nn.Tanh)
        self.bias = nn.Parameter(torch.zeros(1))

        # fixed normalization (stabilizes training; consistent with other scripts)
        self.register_buffer("p_mean", torch.tensor([0.5, 0.5, 0.075], dtype=torch.float32))
        self.register_buffer("p_std",  torch.tensor([0.1, 0.1, 0.025], dtype=torch.float32))

    def forward(self, p, xy):
        assert p.shape[0] == 1, "This implementation assumes one p per call."

        p_norm = (p - self.p_mean.to(p.device)) / self.p_std.to(p.device)
        b = self.branch(p_norm)        # (1, latent)
        t = self.trunk(xy)             # (N, latent)

        u_raw = torch.sum(t * b, dim=1) + self.bias  # (N,)

        x = xy[:, 0:1]
        y = xy[:, 1:2]
        bc_factor = (x * (1.0 - x) * y * (1.0 - y)).view(-1)

        u = u_raw * bc_factor
        return u, None


# ============================================================
# PDE residual: -Δu - f = 0
# ============================================================
def poisson_residual(model, p, xy_int):
    xy_int.requires_grad_(True)
    u, _ = model(p, xy_int)

    grads = torch.autograd.grad(
        u, xy_int, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = grads[:, 0]
    u_y = grads[:, 1]

    grads2_x = torch.autograd.grad(
        u_x, xy_int, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    grads2_y = torch.autograd.grad(
        u_y, xy_int, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0]

    u_xx = grads2_x[:, 0]
    u_yy = grads2_y[:, 1]
    laplace_u = u_xx + u_yy

    x = xy_int[:, 0]
    y = xy_int[:, 1]
    x0, y0, nu = p[0, 0], p[0, 1], p[0, 2]
    f_val = gaussian_source(x, y, x0, y0, nu)

    return -laplace_u - f_val


# ============================================================
# Training loop (unchanged)
# ============================================================
def train_pi_deeponet(
    n_int=4096,
    n_bc=512,
    epochs=3000,
    lr=1e-3,
    latent_dim=128,
    hidden_dim=128,
    depth=3,
    tasks_per_batch=8,
):
    model = ParametricDeepONet(
        latent_dim=latent_dim, hidden_dim=hidden_dim, depth=depth
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=max(epochs // 2, 1), gamma=0.5)

    best_loss = float("inf")
    best_state = None

    history = {"epoch": [], "total_loss": [], "pde_loss": [], "bc_loss": [], "lr": []}

    for ep in range(1, epochs + 1):
        optimizer.zero_grad()

        total_loss = 0.0
        total_pde = 0.0
        total_bc = 0.0

        # ν-curriculum (same as the other methods)
        if ep <= epochs // 2:
            nu_min_curr, nu_max_curr = 0.08, nu_max
        else:
            nu_min_curr, nu_max_curr = nu_min, nu_max

        for _ in range(tasks_per_batch):
            p = sample_pde_param(nu_min_curr, nu_max_curr)

            xy_int, xy_bc = sample_collocation_points_p_dependent(
                n_int, n_bc, p, alpha=None, sigma_factor=3.0, device=device
            )

            res_int = poisson_residual(model, p, xy_int)
            loss_pde = torch.mean(res_int**2)

            # BC monitoring (hard BC already enforced)
            if xy_bc.shape[0] > 0:
                u_bc, _ = model(p, xy_bc)
                loss_bc = torch.mean(u_bc**2)
            else:
                loss_bc = torch.tensor(0.0, device=device)

            loss_task = loss_pde
            total_loss += loss_task
            total_pde  += loss_pde
            total_bc   += loss_bc

        total_loss /= tasks_per_batch
        total_pde  /= tasks_per_batch
        total_bc   /= tasks_per_batch

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        history["epoch"].append(ep)
        history["total_loss"].append(total_loss.item())
        history["pde_loss"].append(total_pde.item())
        history["bc_loss"].append(total_bc.item())
        history["lr"].append(lr_now)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 200 == 0 or ep == 1:
            print(
                f"Epoch {ep:5d} | Loss: {total_loss.item():.3e} "
                f"(PDE: {total_pde.item():.3e}, BC(mon.): {total_bc.item():.3e}, lr={lr_now:.1e})"
            )

    print(f"\nBest training loss = {best_loss:.3e}")
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, history


# ============================================================
# Training curves (optional)
# ============================================================
def plot_training_curves(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["total_loss"], label="Total loss")
    plt.plot(history["epoch"], history["pde_loss"], label="PDE loss", linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.yscale("log")
    plt.title("Training loss vs epoch (PI-DeepONet)")
    plt.legend(); plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["bc_loss"], label="BC monitoring loss")
    plt.xlabel("Epoch"); plt.ylabel("BC loss"); plt.yscale("log")
    plt.title("Boundary-condition loss vs epoch (PI-DeepONet)")
    plt.legend(); plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["lr"])
    plt.xlabel("Epoch"); plt.ylabel("Learning rate")
    plt.title("Learning rate vs epoch (PI-DeepONet)")
    plt.grid(True)
    plt.show()


# ============================================================
# Evaluation + STANDARDIZED EXPORT
# ============================================================
def evaluate_and_plot_solutions(
    model,
    N=60,
    save_data=True,
    out_dir=DATA_DIR,
    export_prefix="pideeponet"
):
    """
    Creates:
      - local figure with rows: [Exact | Pred | |Error|]
      - standardized bundle NPZ for unified plotting (Exact vs KAPI vs HyperPINN vs PI-DeepONet)
      - per-case NPZ files (optional)

    STANDARDIZED bundle keys (ALWAYS present):
      x, y, params, case_names, u_exact, u_pred, abs_error
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    num_tests = len(test_cases)
    fig, axs = plt.subplots(num_tests, 3, figsize=(15, 4 * num_tests), constrained_layout=True)
    if num_tests == 1:
        axs = np.array([axs])

    # Accumulators for export
    case_names = []
    params_list = []
    u_exact_list = []
    u_pred_list = []
    abs_err_list = []

    # Optional extras
    err_list = []
    rel_l2_list = []

    x_vec = None
    y_vec = None

    for i, (name, x0_test, y0_test, nu_test) in enumerate(test_cases):
        print(f"\n=== Test case {i+1}/{num_tests}: {name} ===")
        print(f"  (x0, y0, nu) = ({x0_test:.3f}, {y0_test:.3f}, {nu_test:.3f})")

        # Exact (FDM)
        xg, yg, u_exact = poisson_exact_fdm(N=N, x0=x0_test, y0=y0_test, nu=nu_test)

        # Cache x/y vectors once (assumes same grid across cases)
        if x_vec is None:
            x_vec = np.asarray(xg, dtype=np.float32)
            y_vec = np.asarray(yg, dtype=np.float32)

        # NN evaluation grid
        Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
        xy_grid = np.stack([Xg.reshape(-1), Yg.reshape(-1)], axis=1)
        xy_grid_t = torch.tensor(xy_grid, dtype=torch.float32, device=device)

        p_test = torch.tensor([[x0_test, y0_test, nu_test]], dtype=torch.float32, device=device)

        with torch.no_grad():
            u_pred_flat, _ = model(p_test, xy_grid_t)
        u_pred = u_pred_flat.cpu().numpy().reshape(N, N)

        # Errors
        err = u_pred - u_exact
        abs_err = np.abs(err)

        # Optional metric
        rel_L2 = np.linalg.norm(err.ravel()) / (np.linalg.norm(u_exact.ravel()) + 1e-12)
        print(f"  Relative L2 error = {rel_L2:.3e}")

        # Store
        case_names.append(name)
        params_list.append([x0_test, y0_test, nu_test])
        u_exact_list.append(u_exact.astype(np.float32))
        u_pred_list.append(u_pred.astype(np.float32))
        abs_err_list.append(abs_err.astype(np.float32))

        # Optional extras
        err_list.append(err.astype(np.float32))
        rel_l2_list.append(np.float32(rel_L2))

        # ---- per-case export (debug-friendly) ----
        if save_data:
            safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in name)
            case_path = os.path.join(out_dir, f"{export_prefix}_case_{i:02d}_{safe}.npz")
            np.savez_compressed(
                case_path,
                x=x_vec,
                y=y_vec,
                params=np.asarray([x0_test, y0_test, nu_test], dtype=np.float32),
                case_name=np.asarray(name),
                u_exact=u_exact.astype(np.float32),
                u_pred=u_pred.astype(np.float32),
                abs_error=abs_err.astype(np.float32),

                # optional extras
                error=err.astype(np.float32),
                rel_l2=np.asarray(rel_L2, dtype=np.float32),
                method=np.asarray("PI-DeepONet"),
            )
            print(f"  Saved per-case arrays to: {case_path}")

        # ---- plotting row: exact / pred / abs error ----
        ax0, ax1, ax2 = axs[i, 0], axs[i, 1], axs[i, 2]

        im0 = ax0.imshow(u_exact.T, origin="lower", extent=[0, 1, 0, 1])
        ax0.set_title(r"$u_{\mathrm{exact}}$ (FDM)", fontsize=14)
        ax0.set_xlabel("x"); ax0.set_ylabel("y")
        plt.colorbar(im0, ax=ax0)

        im1 = ax1.imshow(u_pred.T, origin="lower", extent=[0, 1, 0, 1])
        ax1.set_title("u_PI-DeepONet", fontsize=14)
        ax1.set_xlabel("x"); ax1.set_ylabel("y")
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(abs_err.T, origin="lower", extent=[0, 1, 0, 1])
        ax2.set_title("|u_PI-DeepONet - u_exact|", fontsize=14)
        ax2.set_xlabel("x"); ax2.set_ylabel("y")
        plt.colorbar(im2, ax=ax2)

    # Save local figure (3 columns)
    results_path = os.path.join(FIG_DIR, "pideeponet_poisson_results.png")
    plt.savefig(results_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved solution comparison figure to: {results_path}")
    plt.show()

    # ========================================================
    # STANDARDIZED BUNDLE EXPORT (this is what unified plotter loads)
    # ========================================================
    if save_data:
        bundle_path = os.path.join(out_dir, f"{export_prefix}_bundle.npz")

        params_arr   = np.asarray(params_list, dtype=np.float32)          # (n_cases, 3)
        u_exact_arr  = np.stack(u_exact_list, axis=0).astype(np.float32)  # (n_cases, N, N)
        u_pred_arr   = np.stack(u_pred_list, axis=0).astype(np.float32)   # (n_cases, N, N)
        abs_err_arr  = np.stack(abs_err_list, axis=0).astype(np.float32)  # (n_cases, N, N)

        # Optional extras (still safe to include)
        err_arr     = (u_pred_arr - u_exact_arr).astype(np.float32)
        rel_l2_arr  = np.asarray(rel_l2_list, dtype=np.float32)

        np.savez_compressed(
            bundle_path,
            # -------- REQUIRED STANDARD KEYS --------
            x=x_vec,
            y=y_vec,
            params=params_arr,
            case_names=np.asarray(case_names),
            u_exact=u_exact_arr,
            u_pred=u_pred_arr,
            abs_error=abs_err_arr,

            # -------- OPTIONAL EXTRAS --------
            error=err_arr,
            rel_l2=rel_l2_arr,
            method=np.asarray("PI-DeepONet"),
        )

        print(f"\n[Saved standardized bundle] {bundle_path}")
        print("Bundle contains: x, y, params, case_names, u_exact, u_pred, abs_error (+ optional extras)")

        print("\nSummary of relative L2 errors:")
        for nm, e in zip(case_names, rel_l2_arr):
            print(f"  {nm:25s}: {e:.3e}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Defaults consistent with your PI-DeepONet baseline
    n_int = 4096
    n_bc = 512
    epochs = 3000
    lr = 1e-3
    tasks_per_batch = 8

    latent_dim = 128
    hidden_dim = 128
    depth = 3

    # Train
    model, history = train_pi_deeponet(
        n_int=n_int,
        n_bc=n_bc,
        epochs=epochs,
        lr=lr,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        tasks_per_batch=tasks_per_batch,
    )

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Optional curves
    plot_training_curves(history)

    # Reload model (keeps workflow consistent with KAPI/HyperPINN scripts)
    loaded_model = ParametricDeepONet(
        latent_dim=latent_dim, hidden_dim=hidden_dim, depth=depth
    ).to(device)
    loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    loaded_model.eval()
    print("Loaded model from:", MODEL_PATH)

    # Evaluate + standardized export
    evaluate_and_plot_solutions(
        loaded_model,
        N=60,
        save_data=True,
        out_dir=DATA_DIR,
        export_prefix="pideeponet"
    )

    print("\nDone.")
    print("Current directory:", os.getcwd())
    print("Files:", os.listdir("."))
