import numpy as np

CHARGES = {"R":1,"D":-1,"N":0,"E":-1,"K":1,"H":0,"Q":0,"S":0,"C":0,"G":0,
           "T":0,"A":0,"M":0,"Y":0,"V":0,"W":0,"L":0,"I":0,"P":0,"F":0}
LAMBDAS = {"R":0.7307624767517166,"D":0.0416040480605567,"N":0.4255859009787713,
    "E":0.0006935460962935,"K":0.1790211738990582,"H":0.4663667290557992,"Q":0.3934318551056041,
    "S":0.4625416811611541,"C":0.5615435099141777,"G":0.7058843733666401,"T":0.3713162976273964,
    "A":0.2743297969040348,"M":0.5308481134337497,"Y":0.9774611449343455,"V":0.2083769608174481,
    "W":0.9893764740371644,"L":0.6440005007782226,"I":0.5423623610671892,"P":0.3593126576364644,
    "F":0.8672358982062975}
SIGMAS = {"R":0.656,"D":0.558,"N":0.568,"E":0.592,"K":0.636,"H":0.608,"Q":0.602,
    "S":0.518,"C":0.548,"G":0.45,"T":0.562,"A":0.504,"M":0.618,"Y":0.646,"V":0.586,
    "W":0.678,"L":0.618,"I":0.618,"P":0.556,"F":0.636}


def build_residues(sequence, neutralize=False):
    """neutralize=True zeroes every charge -- use this to build a 'tailless'
    reference sequence (no charge for the DNA-attraction term to act on),
    since there's no literal way to simulate 'no chain' in a single-chain
    model. Swap in whatever operational definition you actually intend --
    this is a reasonable default, not the only valid one."""
    out = []
    for a in sequence:
        q = 0 if neutralize else CHARGES.get(a, 0)
        out.append({"AA": a, "charge": q, "lambda": LAMBDAS.get(a, 0.0), "sigma": SIGMAS.get(a, 0.0)})
    return out


def initialize_chain(residues, bond_length=0.38, core_radius=4.25, min_sep_factor=0.85):
    N = len(residues)
    positions = np.zeros((N, 3))
    positions[0] = [core_radius + 0.01, 0.0, 0.0]
    for i in range(1, N):
        placed = False
        for _ in range(2000):
            theta = np.random.uniform(0, np.pi * 2)
            phi = np.random.uniform(0, np.pi)
            direction = np.array([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)])
            candidate = positions[i - 1] + bond_length * direction
            if np.linalg.norm(candidate) < core_radius:
                continue
            ok = True
            for j in range(i - 1):
                min_sep = min_sep_factor * 0.5 * (residues[i]["sigma"] + residues[j]["sigma"])
                if np.linalg.norm(candidate - positions[j]) < min_sep:
                    ok = False
                    break
            if ok:
                positions[i] = candidate
                placed = True
                break
        if not placed:
            positions[i] = candidate
    return positions


def calculate_energy(positions, residues, core_radius=4.25, bond_length=0.38, bond_force=507.866,
                      l_B=0.7, kappa=1.2, dna_charge=-30, dna_attraction_scale=0.05,
                      dna_position=(0.0, 0.0, 0.0), epsilon=0.8368, r_c=2.0, convert_to_kT=True):
    epsilon_kT = epsilon / (0.008314 * 298.15) if convert_to_kT else epsilon
    N = len(positions)
    dna_position = np.asarray(dna_position, dtype=float)

    for b in positions:
        if np.linalg.norm(b) < core_radius:
            return float("inf")

    energy = 0.0
    for i in range(1, N):
        d = np.linalg.norm(positions[i] - positions[i - 1])
        energy += 0.5 * bond_force * (d - bond_length) ** 2

    lambdas = np.array([r["lambda"] for r in residues])
    sigmas = np.array([r["sigma"] for r in residues])
    lambda_ij = 0.5 * (lambdas[:, None] + lambdas[None, :])
    sigma_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
    dist_all = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    ii, jj = np.triu_indices(N, k=1)
    nonbonded = (jj - ii) > 1
    ii, jj = ii[nonbonded], jj[nonbonded]
    r = dist_all[ii, jj]
    l_ij = lambda_ij[ii, jj]
    s_ij = sigma_ij[ii, jj]
    r_safe = np.where(r < 0.3, 0.3, r)
    s_over_r, s_over_rc = s_ij / r_safe, s_ij / r_c
    lj_r = 4 * epsilon_kT * (s_over_r**12 - s_over_r**6)
    lj_rc = 4 * epsilon_kT * (s_over_rc**12 - s_over_rc**6)
    r_inf = (2 ** (1/6)) * s_ij
    core_mask = r <= r_inf
    tail_mask = (r > r_inf) & (r < r_c)
    energy += np.sum(lj_r[core_mask] - l_ij[core_mask]*lj_rc[core_mask] + epsilon_kT*(1 - l_ij[core_mask]))
    energy += np.sum(l_ij[tail_mask] * (lj_r[tail_mask] - lj_rc[tail_mask]))

    for i in range(N):
        for j in range(i + 1, N):
            qi, qj = residues[i]["charge"], residues[j]["charge"]
            if qi == 0 or qj == 0:
                continue
            d = np.linalg.norm(positions[i] - positions[j])
            d = max(d, 0.001)
            energy += l_B * qi * qj * np.exp(-kappa * d) / d

    for i in range(N):
        qi = residues[i]["charge"]
        if qi <= 0:
            continue
        d_center = np.linalg.norm(positions[i] - dna_position)
        d_surface = max(d_center - core_radius, 0.30) 
        energy += dna_attraction_scale * l_B * qi * dna_charge * np.exp(-kappa * d_surface) / d_surface

    return energy


def monte_carlo_step(positions, residues, step_size=0.08, **kwargs):
    N = len(positions)
    i = np.random.randint(0, N)
    old_pos = positions[i].copy()
    old_e = calculate_energy(positions, residues, **kwargs)
    positions[i] += np.random.normal(0, step_size, 3)
    new_e = calculate_energy(positions, residues, **kwargs)
    delta_E = new_e - old_e
    accepted = True
    if delta_E > 0:
        if np.random.random() >= np.exp(-delta_E):
            positions[i] = old_pos
            accepted = False
    return accepted


def run_simulation(sequence, n_equilibration=20000, n_production=50000, sample_interval=100,
                    step_size=0.08, seed=None, check_every=200, neutralize=False, **kwargs):
    if seed is not None:
        np.random.seed(seed)
    residues = build_residues(sequence, neutralize=neutralize)
    positions = initialize_chain(residues, core_radius=kwargs.get("core_radius", 4.25),
                                  bond_length=kwargs.get("bond_length", 0.38))
    dna_position = np.asarray(kwargs.get("dna_position", (0.0, 0.0, 0.0)), dtype=float)
    core_radius = kwargs.get("core_radius", 4.25)
    charged_idx = [i for i, r in enumerate(residues) if r["charge"] > 0]

    for step in range(n_equilibration):
        monte_carlo_step(positions, residues, step_size=step_size, **kwargs)

    Rg_list, dist_list, charged_dist_list = [], [], []
    for step in range(n_production):
        monte_carlo_step(positions, residues, step_size=step_size, **kwargs)
        if step % sample_interval == 0:
            c = positions.mean(axis=0)
            Rg_list.append(np.sqrt(np.mean(np.sum((positions - c)**2, axis=1))))
            dist_list.append(np.mean([np.linalg.norm(p - dna_position) for p in positions]))
            if charged_idx:
                charged_dist_list.append(np.mean(
                    [np.linalg.norm(positions[i]-dna_position) - core_radius for i in charged_idx]))

    return {"Rg_mean": float(np.mean(Rg_list)), "Rg_sem": float(np.std(Rg_list, ddof=1)/np.sqrt(len(Rg_list))),
            "charged_dist_mean": float(np.mean(charged_dist_list)) if charged_dist_list else float("nan"),
            "charged_dist_sem": float(np.std(charged_dist_list, ddof=1)/np.sqrt(len(charged_dist_list))) if charged_dist_list else float("nan")}


def run_replicates(sequence, n_replicates=5, seeds=None, **kwargs):
    seeds = seeds or list(range(n_replicates))
    runs = [run_simulation(sequence, seed=s, **kwargs) for s in seeds]
    rg = np.array([r["Rg_mean"] for r in runs])
    dist = np.array([r["dist_mean"] for r in runs])
    n = len(seeds)
    return {
        "sequence": sequence, "n_replicates": n, "per_replicate": runs,
        "Rg_mean_of_means": float(rg.mean()),
        "Rg_sem": float(rg.std(ddof=1)/np.sqrt(n)) if n > 1 else float("nan"),
        "dist_mean_of_means": float(dist.mean()),
        "dist_sem": float(dist.std(ddof=1)/np.sqrt(n)) if n > 1 else float("nan"),
    }


def calibrate_dna_scale(wt_sequence, scale_values, n_equilibration=8000, n_production=4000,
                            n_replicates=5, **kwargs):
    for scale in scale_values:
        on  = [run_simulation(wt_sequence, seed=s, dna_attraction_scale=scale,
                               n_equilibration=n_equilibration, n_production=n_production, **kwargs)
               for s in range(n_replicates)]
        off = [run_simulation(wt_sequence, seed=s, dna_attraction_scale=0.0,
                               n_equilibration=n_equilibration, n_production=n_production, **kwargs)
               for s in range(n_replicates)]
        for label, key in [("Rg", "Rg_mean"), ("charged_dist", "charged_dist_mean")]:
            on_vals = np.array([r[key] for r in on]); off_vals = np.array([r[key] for r in off])
            gap = off_vals.mean() - on_vals.mean()
            noise = np.sqrt(on_vals.std(ddof=1)**2 + off_vals.std(ddof=1)**2) / np.sqrt(n_replicates)
            print(f"scale={scale:.3f}  {label}: on={on_vals.mean():.3f} off={off_vals.mean():.3f} "
                  f"gap={gap:.3f} gap/noise={gap/noise if noise>0 else float('nan'):.2f}")


if __name__ == "__main__":
    wt = "ARTKQTARKSTGGKAPRKQLATKAARKS"
    np.random.seed(0)
    residues = build_residues(wt)
    positions = initialize_chain(residues)
    nb = [np.linalg.norm(positions[i]-positions[j]) for i in range(len(positions)) for j in range(i+2, len(positions))]
    print("self-check -- min nonbonded distance:", min(nb), "(should be roughly 0.4-0.6, not ~0.1)")
    print("self-check -- starting energy:", calculate_energy(positions, residues),
          "(should be tens of kT, not tens of thousands)")

wt = "ARTKQTARKSTGGKAPRKQLATKAARKS"
results = calibrate_dna_scale(
    wt,
    scale_values=[0.2, 0.5, 1.0], 
    n_equilibration=20000,
    n_production=10000,
    n_replicates=5,
)

from scipy.stats import spearmanr

VALIDATION = [
    ("unmodified",        50,  "ARTKQTARKSTGGKAPRKQLATKAARKSAPATGGVKKPHRYRPG"),
    ("R2/8Q",              100, "AQTKQTAQKSTGGKAPRKQLATKAARKSAPATGGVKKPHRYRPG"),
    ("R17/26Q",             50, "ARTKQTARKSTGGKAPQKQLATKAAQKSAPATGGVKKPHRYRPG"),
    ("R2/8/17/26Q",        200, "AQTKQTAQKSTGGKAPQKQLATKAAQKSAPATGGVKKPHRYRPG"),
    ("K4/9Q",              100, "ARTQQTARQSTGGKAPRKQLATKAARKSAPATGGVKKPHRYRPG"),
    ("K4/9/18/27Q",        250, "ARTQQTARQSTGGKAPRQQLATKAARQSAPATGGVKKPHRYRPG"),
    ("K4/9/14/18/23/27Q",  350, "ARTQQTARQSTGGQAPRQQLATQAARQSAPATGGVKKPHRYRPG"),
    ("R2/8/17/26K",         15, "AKTKQTAKKSTGGKAPKKQLATKAAKKSAPATGGVKKPHRYRPG"),
    ("K4/9/14/18/23/27R",   35, "ARTRQTARRSTGGRAPRRQLATRAARRSAPATGGVKKPHRYRPG"),
]

def validate_scale(scale, n_equilibration=8000, n_production=4000, n_replicates=3, **kwargs):
    csats, charged_dists = [], []
    for name, csat, seq in VALIDATION:
        runs = [run_simulation(seq, seed=s, dna_attraction_scale=scale,
                                n_equilibration=n_equilibration, n_production=n_production, **kwargs)
                for s in range(n_replicates)]
        cd = np.mean([r["charged_dist_mean"] for r in runs])
        charged_dists.append(cd)
        csats.append(csat)
        print(f"  {name:22s} charged_dist={cd:.3f}")
    rho, p = spearmanr(csats, charged_dists)
    print(f"scale={scale}: Spearman rho={rho:.3f}, p={p:.3f}\n")
    return rho, p

for scale in [0.2, 0.5, 1.0, 2.0]:
    validate_scale(scale)

def validate_scale(scale, n_equilibration=8000, n_production=4000, n_replicates=3, **kwargs):
    csats, charged_dists, rgs = [], [], []
    for name, csat, seq in VALIDATION:
        runs = [run_simulation(seq, seed=s, dna_attraction_scale=scale,
                                n_equilibration=n_equilibration, n_production=n_production, **kwargs)
                for s in range(n_replicates)]
        cd = np.mean([r["charged_dist_mean"] for r in runs])
        rg = np.mean([r["Rg_mean"] for r in runs]) 
        charged_dists.append(cd)
        rgs.append(rg) 
        csats.append(csat)
        print(f"  {name:22s} charged_dist={cd:.3f}  Rg={rg:.3f}")
    rho_cd, p_cd = spearmanr(csats, charged_dists)
    rho_rg, p_rg = spearmanr(csats, rgs)  
    print(f"scale={scale}: charged_dist rho={rho_cd:.3f} p={p_cd:.3f}  |  Rg rho={rho_rg:.3f} p={p_rg:.3f}\n")
    return rho_cd, p_cd, rho_rg, p_rg

validate_scale(0.2)
