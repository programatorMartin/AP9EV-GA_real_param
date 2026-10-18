import numpy as np
import matplotlib.pyplot as plt

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

def schwefel(x: np.ndarray) -> float:
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def decode_fixed_point(bits: np.ndarray, bounds: tuple) -> float:
    integer_val = int("".join(bits.astype(str)), 2)
    max_int = (1 << 32) - 1
    return bounds[0] + (bounds[1] - bounds[0]) * (integer_val / max_int)

def real_mutation_normal(individual: np.ndarray, mutation_rate: float, sigma: float = 1.0) -> np.ndarray:
    mutated = individual.copy()
    mask = np.random.rand(len(individual)) < mutation_rate
    mutated[mask] += np.random.normal(0, sigma, size=np.sum(mask))
    return mutated

def real_mutation_uniform(individual: np.ndarray, mutation_rate: float, bounds: tuple) -> np.ndarray:
    mutated = individual.copy()
    mask = np.random.rand(len(individual)) < mutation_rate
    mutated[mask] = np.random.uniform(bounds[0], bounds[1], size=np.sum(mask))
    return mutated

def onemax(individual: np.ndarray) -> int:
    return int(np.sum(individual))

def leading_ones(individual: np.ndarray) -> int:
    count = 0
    for bit in individual:
        if bit == 1:
            count += 1
        else:
            break
    return count

def roulette_wheel_selection(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    total_fit = np.sum(fitness)
    if total_fit == 0:
        probs = np.ones(len(fitness)) / len(fitness)
    else:
        probs = fitness / total_fit
    idx = np.random.choice(len(population), p=probs)
    return population[idx].copy()

def rank_selection(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    n = len(fitness)
    ranks = np.argsort(np.argsort(fitness)) + 1
    probs = ranks / np.sum(ranks)
    idx = np.random.choice(n, p=probs)
    return population[idx].copy()

def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float = 0.9):
    d = len(parent1)
    if np.random.rand() < crossover_rate and d > 1:
        point = np.random.randint(1, d)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def bit_flip_mutation(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    mutation_mask = np.random.rand(len(individual)) < mutation_rate
    mutated = individual.copy()
    mutated[mutation_mask] = 1 - mutated[mutation_mask]
    return mutated

def run_ga(
    fitness_fn,
    dim: int,
    pop_size: int = 50,
    elitism_ratio: float = 0.15,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.01,
    selection_type: str = "rank",
    max_evaluations: int = 1000,
):
    pop = np.random.randint(0, 2, size=(pop_size, dim))
    fitness = np.array([fitness_fn(ind) for ind in pop])
    eval_count = pop_size

    current_best = int(np.max(fitness))
    eval_history = [eval_count]
    fitness_history = [current_best]

    n_elites = max(1, int(round(pop_size * elitism_ratio)))
    selection_fn = rank_selection if selection_type == "rank" else roulette_wheel_selection

    while eval_count < max_evaluations:
        sorted_indices = np.argsort(fitness)[::-1]

        new_pop = [pop[idx].copy() for idx in sorted_indices[:n_elites]]
        new_fitness = [int(fitness[idx]) for idx in sorted_indices[:n_elites]]

        while len(new_pop) < pop_size and eval_count < max_evaluations:
            p1 = selection_fn(pop, fitness)
            p2 = selection_fn(pop, fitness)
            c1, c2 = one_point_crossover(p1, p2, crossover_rate)

            c1 = bit_flip_mutation(c1, mutation_rate)
            new_pop.append(c1)
            new_fitness.append(fitness_fn(c1))
            eval_count += 1

            if len(new_pop) < pop_size and eval_count < max_evaluations:
                c2 = bit_flip_mutation(c2, mutation_rate)
                new_pop.append(c2)
                new_fitness.append(fitness_fn(c2))
                eval_count += 1

        pop = np.array(new_pop)
        fitness = np.array(new_fitness)

        current_best = int(np.max(fitness))
        eval_history.append(eval_count)
        fitness_history.append(current_best)

    return current_best, np.array(eval_history), np.array(fitness_history)

def run_benchmark():
    dimensions = [10, 30, 100]
    n_runs = 10
    problems = [("One-Max", onemax), ("Leading-Ones", leading_ones)]

    fig, axes = plt.subplots(len(problems), len(dimensions), figsize=(15, 8), sharey=False)
    fig.suptitle("Priemerná konvergencia GA za 10 nezávislých behov", fontsize=14, fontweight="bold")

    for row_idx, (p_name, p_fn) in enumerate(problems):
        print(f"\n{'='*70}\nPROBLÉM: {p_name}\n{'='*70}")

        for col_idx, D in enumerate(dimensions):
            max_evals = 100 * D
            pop_size = 30 if D == 10 else (50 if D == 30 else 80)
            elitism_ratio = 0.15
            crossover_rate = 0.9
            mutation_rate = 0.01
            selection_type = "rank"

            final_results = []
            grid_evals = np.linspace(pop_size, max_evals, num=100)
            interpolated_curves = []

            for run in range(n_runs):
                best_fit, evals, fits = run_ga(
                    fitness_fn=p_fn,
                    dim=D,
                    pop_size=pop_size,
                    elitism_ratio=elitism_ratio,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    selection_type=selection_type,
                    max_evaluations=max_evals,
                )
                final_results.append(best_fit)

                interp_fit = np.interp(grid_evals, evals, fits)
                interpolated_curves.append(interp_fit)

            best_val = int(np.max(final_results))
            worst_val = int(np.min(final_results))
            mean_val = float(np.mean(final_results))
            median_val = float(np.median(final_results))
            std_val = float(np.std(final_results))

            print(
                f"Dim: {D:3d} (Max Evals: {max_evals:5d}) | "
                f"Best: {best_val:3d}/{D} | Worst: {worst_val:3d}/{D} | "
                f"Mean: {mean_val:5.2f} | Median: {median_val:5.2f} | Std: {std_val:4.2f}"
            )

            mean_curve = np.mean(interpolated_curves, axis=0)
            ax = axes[row_idx, col_idx]
            ax.plot(grid_evals, mean_curve, color="navy", lw=2, label="Priemerné best fitness")
            ax.axhline(y=D, color="crimson", linestyle="--", alpha=0.7, label=f"Optimum ({D})")
            ax.set_title(f"{p_name} | {D}D (Evals: {max_evals})")
            ax.set_xlabel("Počet evaluácií")
            ax.set_ylabel("Najlepšie fitness")
            ax.grid(True, linestyle=":", alpha=0.6)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    run_benchmark()