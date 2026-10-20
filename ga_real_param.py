import numpy as np
import struct
import matplotlib.pyplot as plt

def sphere(x):
    return float(np.sum(x**2))

def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

def schwefel(x):
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def decode_ieee754(bits, bounds):
    b_str = "".join(bits.astype(str))
    int_val = int(b_str, 2)
    try:
        val = struct.unpack('>f', struct.pack('>I', int_val))[0]
        if np.isnan(val) or np.isinf(val):
            val = bounds[0]
    except Exception:
        val = bounds[0]
    return float(np.clip(val, bounds[0], bounds[1]))

def decode_fixed_point(bits, bounds):
    # 1 bit znamienko | n_int bitov celá časť | n_frac bitov desatinná časť (spolu 32 bitov)
    # n_int sa odvodí z rozsahu úlohy a je pevné, n_frac = 31 - n_int
    max_abs = max(abs(bounds[0]), abs(bounds[1]))
    n_int = int(np.ceil(np.log2(max_abs + 1)))
    n_frac = 31 - n_int
    b_str = "".join(bits.astype(str))
    sign = -1.0 if b_str[0] == "1" else 1.0
    int_part = int(b_str[1:1 + n_int], 2)
    frac_part = int(b_str[1 + n_int:], 2) / float(1 << n_frac)
    return float(np.clip(sign * (int_part + frac_part), bounds[0], bounds[1]))

def decode_bcd(bits, bounds):
    val_str = ""
    for i in range(0, len(bits), 4):
        nibble = bits[i:i+4]
        n_int = int("".join(nibble.astype(str)), 2) % 10
        val_str += str(n_int)
    frac = float("0." + val_str)
    return float(bounds[0] + (bounds[1] - bounds[0]) * frac)

def decode_individual_binary(ind, dim, bounds, method):
    x = np.zeros(dim)
    for i in range(dim):
        bits = ind[i*32:(i+1)*32]
        if method == "ieee754":
            x[i] = decode_ieee754(bits, bounds)
        elif method == "fixed":
            x[i] = decode_fixed_point(bits, bounds)
        elif method == "bcd":
            x[i] = decode_bcd(bits, bounds)
    return x

def get_rank_probs(fitness):
    ranks = np.argsort(np.argsort(-fitness)) + 1
    return ranks / np.sum(ranks)

def one_point_crossover(p1, p2, crossover_rate=0.9):
    d = len(p1)
    if np.random.rand() < crossover_rate and d > 1:
        point = np.random.randint(1, d)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2
    return p1.copy(), p2.copy()

def real_mutation_normal(ind, mutation_rate, sigma, bounds):
    mutated = ind.copy()
    mask = np.random.rand(len(ind)) < mutation_rate
    mutated[mask] += np.random.normal(0, sigma, size=np.sum(mask))
    return np.clip(mutated, bounds[0], bounds[1])

def real_mutation_uniform(ind, mutation_rate, bounds):
    mutated = ind.copy()
    mask = np.random.rand(len(ind)) < mutation_rate
    mutated[mask] = np.random.uniform(bounds[0], bounds[1], size=np.sum(mask))
    return mutated

def bit_flip_mutation(ind, mutation_rate):
    mask = np.random.rand(len(ind)) < mutation_rate
    mutated = ind.copy()
    mutated[mask] = 1 - mutated[mask]
    return mutated

def run_ga_binary(fitness_fn, dim, bounds, method, pop_s=50, max_ev=10000):
    ind_len = dim * 32
    pop = np.random.randint(0, 2, size=(pop_s, ind_len))
    fitness = np.array([fitness_fn(decode_individual_binary(ind, dim, bounds, method)) for ind in pop])
    evals = pop_s
    best_fit = float(np.min(fitness))
    
    eval_history = [evals]
    fit_history = [best_fit]
    n_elites = max(1, int(round(0.15 * pop_s)))
    
    while evals < max_ev:
        new_pop = []
        new_fit = []
        elite_idx = np.argsort(fitness)[:n_elites]
        for idx in elite_idx:
            new_pop.append(pop[idx].copy())
            new_fit.append(fitness[idx])
        
        probs = get_rank_probs(fitness)
        
        while len(new_pop) < pop_s and evals < max_ev:
            p1_idx = np.random.choice(pop_s, p=probs)
            p2_idx = np.random.choice(pop_s, p=probs)
            p1, p2 = pop[p1_idx], pop[p2_idx]
            c1, c2 = one_point_crossover(p1, p2, 0.9)
            
            c1 = bit_flip_mutation(c1, 0.01)
            f1 = fitness_fn(decode_individual_binary(c1, dim, bounds, method))
            new_pop.append(c1)
            new_fit.append(f1)
            evals += 1
            
            if len(new_pop) < pop_s and evals < max_ev:
                c2 = bit_flip_mutation(c2, 0.01)
                f2 = fitness_fn(decode_individual_binary(c2, dim, bounds, method))
                new_pop.append(c2)
                new_fit.append(f2)
                evals += 1
                
        pop = np.array(new_pop)
        fitness = np.array(new_fit)
        curr_best = float(np.min(fitness))
        if curr_best < best_fit:
            best_fit = curr_best
        eval_history.append(evals)
        fit_history.append(best_fit)
        
    return best_fit, np.array(eval_history), np.array(fit_history)

def run_ga_real(fitness_fn, dim, bounds, mutation_type, sigma=1.0, mut_rate=0.1, pop_s=50, max_ev=10000):
    pop = np.random.uniform(bounds[0], bounds[1], size=(pop_s, dim))
    fitness = np.array([fitness_fn(ind) for ind in pop])
    evals = pop_s
    best_fit = float(np.min(fitness))
    
    eval_history = [evals]
    fit_history = [best_fit]
    n_elites = max(1, int(round(0.15 * pop_s)))
    
    while evals < max_ev:
        new_pop = []
        new_fit = []
        elite_idx = np.argsort(fitness)[:n_elites]
        for idx in elite_idx:
            new_pop.append(pop[idx].copy())
            new_fit.append(fitness[idx])
        
        probs = get_rank_probs(fitness)
        
        while len(new_pop) < pop_s and evals < max_ev:
            p1_idx = np.random.choice(pop_s, p=probs)
            p2_idx = np.random.choice(pop_s, p=probs)
            p1, p2 = pop[p1_idx], pop[p2_idx]
            c1, c2 = one_point_crossover(p1, p2, 0.9)
            
            if mutation_type == "normal":
                c1 = real_mutation_normal(c1, mut_rate, sigma, bounds)
            else:
                c1 = real_mutation_uniform(c1, mut_rate, bounds)
            f1 = fitness_fn(c1)
            new_pop.append(c1)
            new_fit.append(f1)
            evals += 1
            
            if len(new_pop) < pop_s and evals < max_ev:
                if mutation_type == "normal":
                    c2 = real_mutation_normal(c2, mut_rate, sigma, bounds)
                else:
                    c2 = real_mutation_uniform(c2, mut_rate, bounds)
                f2 = fitness_fn(c2)
                new_pop.append(c2)
                new_fit.append(f2)
                evals += 1
                
        pop = np.array(new_pop)
        fitness = np.array(new_fit)
        curr_best = float(np.min(fitness))
        if curr_best < best_fit:
            best_fit = curr_best
        eval_history.append(evals)
        fit_history.append(best_fit)
        
    return best_fit, np.array(eval_history), np.array(fit_history)

def make_variants(pop_s, max_ev, sigma_frac, mut_rate):
    return [
        ("IEEE 754", lambda fn, d, b: run_ga_binary(fn, d, b, "ieee754", pop_s=pop_s, max_ev=max_ev)),
        ("Fixed-Point", lambda fn, d, b: run_ga_binary(fn, d, b, "fixed", pop_s=pop_s, max_ev=max_ev)),
        ("BCD", lambda fn, d, b: run_ga_binary(fn, d, b, "bcd", pop_s=pop_s, max_ev=max_ev)),
        ("Real Normal", lambda fn, d, b: run_ga_real(fn, d, b, "normal", sigma=sigma_frac*(b[1]-b[0]), mut_rate=mut_rate, pop_s=pop_s, max_ev=max_ev)),
        ("Real Uniform", lambda fn, d, b: run_ga_real(fn, d, b, "uniform", mut_rate=mut_rate_uniform, pop_s=pop_s, max_ev=max_ev))
    ]

def tune_normal_mutation(p_name, p_fn, bounds, dim, pop_s, max_ev, tune_runs=5):
    # Experiment so sigma (ako zlomok rozsahu) a pravdepodobnosťou mutácie pre Real Normal
    sigma_fracs = [0.01, 0.05, 0.2]
    mut_rates = [0.05, 0.1, 0.3]
    print(f"\n--- Ladenie Real Normal na {p_name} (priemer z {tune_runs} behov) ---")
    print(f"{'sigma/rozsah':>13s} | {'mut_rate':>8s} | {'Mean':>12s}")
    best = (None, None, np.inf)
    for sf in sigma_fracs:
        for mr in mut_rates:
            res = [run_ga_real(p_fn, dim, bounds, "normal", sigma=sf*(bounds[1]-bounds[0]),
                               mut_rate=mr, pop_s=pop_s, max_ev=max_ev)[0] for _ in range(tune_runs)]
            m = float(np.mean(res))
            print(f"{sf:13.2f} | {mr:8.2f} | {m:12.4f}")
            if m < best[2]:
                best = (sf, mr, m)
    print(f"-> Vybrané: sigma = {best[0]} * rozsah, mut_rate = {best[1]}")
    return best[0], best[1]

mut_rate_uniform = 0.1

def main():
    np.random.seed(42)
    dim = 10
    max_ev = 10000
    pop_s = 50
    n_runs = 10
    
    problems = [
        ("Sphere", sphere, (-5.12, 5.12)),
        ("Rosenbrock", rosenbrock, (-2.048, 2.048)),
        ("Schwefel", schwefel, (-500.0, 500.0))
    ]
    
    fig, axes = plt.subplots(1, len(problems), figsize=(18, 5))
    fig.suptitle("Priemerná konvergencia GA za 10 nezávislých behov (10D)", fontsize=14, fontweight="bold")
    grid_evals = np.linspace(pop_s, max_ev, num=100)
    
    for col_idx, (p_name, p_fn, bounds) in enumerate(problems):
        sigma_frac, mut_rate = tune_normal_mutation(p_name, p_fn, bounds, dim, pop_s, max_ev)
        variants = make_variants(pop_s, max_ev, sigma_frac, mut_rate)
        
        print(f"\n{'='*80}\nPROBLÉM: {p_name} (10D)\n{'='*80}")
        ax = axes[col_idx]
        
        for v_name, v_fn in variants:
            final_results = []
            interp_curves = []
            
            for _ in range(n_runs):
                best_f, evals, fits = v_fn(p_fn, dim, bounds)
                final_results.append(best_f)
                interp_curves.append(np.interp(grid_evals, evals, fits))
                
            best_val = np.min(final_results)
            worst_val = np.max(final_results)
            mean_val = np.mean(final_results)
            median_val = np.median(final_results)
            std_val = np.std(final_results)
            
            print(
                f"{v_name:14s} | Best: {best_val:10.4f} | Worst: {worst_val:10.4f} | "
                f"Mean: {mean_val:10.4f} | Median: {median_val:10.4f} | Std: {std_val:10.4f}"
            )
            
            mean_curve = np.mean(interp_curves, axis=0)
            ax.plot(grid_evals, mean_curve, lw=2, label=v_name)
            
        ax.set_title(f"{p_name} | 10D (Evals: {max_ev})")
        ax.set_xlabel("Počet evaluácií")
        ax.set_ylabel("Najlepšie fitness (log škála)")
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")
        
    plt.tight_layout()
    plt.savefig("benchmark_real_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
