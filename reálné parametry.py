import numpy as np
import matplotlib.pyplot as plt

D = 10
MAX_EVALS = 10_000
RUNS = 10
POP_SIZE = 50
ELITE_RATIO = 0.1
CROSSOVER = 1.0
BIT_MUTATION = 1 / (32 * D)  


def sphere(x):
    return float(np.sum(x * x))


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def schwefel(x):
    return float(418.9828872724338 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


FUNCTIONS = {
    'Sphere': (sphere, -5.12, 5.12),
    'Rosenbrock': (rosenbrock, -5.0, 10.0),
    'Schwefel': (schwefel, -500.0, 500.0),
}
SETTINGS = {
    'IEEE 754': dict(kind='ieee'),
    'Fixed point': dict(kind='fixed'),
    'BCD': dict(kind='bcd'),
    'Normal s=1% p=0.1': dict(kind='normal', sigma_ratio=.01, mutation=.1),
    'Normal s=1% p=0.5': dict(kind='normal', sigma_ratio=.01, mutation=.5),
    'Normal s=10% p=0.1': dict(kind='normal', sigma_ratio=.1, mutation=.1),
    'Normal s=10% p=0.5': dict(kind='normal', sigma_ratio=.1, mutation=.5),
    'Uniform p=0.1': dict(kind='uniform', mutation=.1),
}
BIT_WEIGHTS = 2 ** np.arange(31, -1, -1, dtype=np.uint64)
DIGIT_WEIGHTS = 10 ** np.arange(6, -1, -1, dtype=np.int64)


def encode(x, kind):
    if kind == 'ieee':
        words = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    elif kind == 'fixed':
        words = (np.rint(x * 2**22).astype(np.int64) & 0xffffffff).astype(np.uint64)
    elif kind == 'bcd':
        digits = np.rint(np.abs(x) * 10_000).astype(np.int64)
        words = np.zeros(len(x), dtype=np.uint64)
        for power in DIGIT_WEIGHTS:
            words = (words << np.uint64(4)) | ((digits // power) % 10).astype(np.uint64)
        words = (words << np.uint64(4)) | np.where(x < 0, 0xD, 0xC).astype(np.uint64)
    else:
        raise ValueError(kind)
    return ((words[:, None] & BIT_WEIGHTS) != 0).astype(np.uint8).ravel()


def decode(bits, kind):
    words = np.sum(bits.reshape(D, 32) * BIT_WEIGHTS, axis=1, dtype=np.uint64)
    if kind == 'ieee':
        with np.errstate(invalid='ignore'):
            return words.astype(np.uint32).view(np.float32).astype(float)
    if kind == 'fixed':
        signed = words.astype(np.int64)
        signed = np.where(signed >= 2**31, signed - 2**32, signed)
        return signed / 2**22
    sign = words & 15
    digits = ((words[:, None] >> np.arange(28, 0, -4, dtype=np.uint64)) & 15)
    if np.any(digits > 9) or np.any((sign != 0xC) & (sign != 0xD)):
        return None
    return np.sum(digits * DIGIT_WEIGHTS, axis=1) / 10_000 * np.where(sign == 0xD, -1, 1)


def genetic_algorithm(fitness, low, high, kind, seed, sigma_ratio=.01, mutation=.1,
                      initial=None):
    rng = np.random.default_rng(seed)
    binary = kind in ('ieee', 'fixed', 'bcd')
    initial = rng.uniform(low, high, (POP_SIZE, D)) if initial is None else initial
    population = np.array([encode(x, kind) for x in initial]) if binary else initial.copy()
    history = []
    best = float('inf')
    invalid = 0

    def evaluate(individual):
        nonlocal best, invalid
        x = decode(individual, kind) if binary else individual
        if x is None or not np.all(np.isfinite(x)) or np.any(x < low) or np.any(x > high):
            score = float('inf')
            invalid += 1
        else:
            score = fitness(x)
        best = min(best, score)
        history.append(best)
        return score

    scores = np.array([evaluate(x) for x in population])
    elite_count = int(POP_SIZE * ELITE_RATIO)
    while len(history) < MAX_EVALS:
        order = np.argsort(scores, kind='stable')
        new_population = [population[i].copy() for i in order[:elite_count]]
        new_scores = list(scores[order[:elite_count]])
        _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
        starts = np.cumsum(counts) - counts
        ranks = starts + (counts + 1) / 2
        weights = POP_SIZE + 1 - ranks[inverse]
        weights /= weights.sum()
        while len(new_population) < POP_SIZE and len(history) < MAX_EVALS:
            a, b = rng.choice(POP_SIZE, size=2, p=weights)
            p1, p2 = population[a], population[b]
            if rng.random() < CROSSOVER:
                point = int(rng.integers(1, len(p1)))
                children = [np.concatenate((p1[:point], p2[point:])),
                            np.concatenate((p2[:point], p1[point:]))]
            else:
                children = [p1.copy(), p2.copy()]
            for child in children:
                if len(new_population) == POP_SIZE or len(history) == MAX_EVALS:
                    break
                if binary:
                    child ^= (rng.random(len(child)) < BIT_MUTATION).astype(np.uint8)
                else:
                    mask = rng.random(D) < mutation  
                    if kind == 'normal':
                        child[mask] += rng.normal(0, sigma_ratio * (high-low), mask.sum())
                        child = np.clip(child, low, high)
                    else:
                        child[mask] = rng.uniform(low, high, mask.sum())
                new_scores.append(evaluate(child))
                new_population.append(child)
        population, scores = np.array(new_population), np.array(new_scores)
    return best, np.array(history), invalid


def main():
    print("Populacia 50, elitizmus 10 %, krizenie 100 %, poradova selekcia.")
    print("Kazda funkcia: 10D, 10 000 ohodnoteni, 10 behov.")
    print("Minimalizacia - mensia hodnota je lepsia.")
    print("Std = populacna smerodajna odchylka (ddof=0).")

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    for ax, (name, (fitness, low, high)) in zip(axes, FUNCTIONS.items()):
        comparison = []
        print(f"\n{name}")
        print(f"{'Nastavenie':<25} {'Best':>12} {'Worst':>12} "
              f"{'Mean':>12} {'Median':>12} {'Std':>12}")

        for label, parameters in SETTINGS.items():
            results, histories = [], []

            for run in range(RUNS):
                rng = np.random.default_rng(1000 + run)
                initial = rng.uniform(low, high, (POP_SIZE, D))
                result, history, _ = genetic_algorithm(
                    fitness, low, high, seed=2000 + run,
                    initial=initial, **parameters)
                results.append(result)
                histories.append(history)

            mean = np.mean(results)
            comparison.append((mean, label))
            print(f"{label:<25} {min(results):>12.5g} {max(results):>12.5g} "
                  f"{mean:>12.5g} {np.median(results):>12.5g} "
                  f"{np.std(results):>12.5g}", flush=True)

            avg_history = np.mean(histories, axis=0)
            ax.plot(np.arange(1, MAX_EVALS + 1), avg_history, label=label)

        winner = min(comparison, key=lambda item: item[0])
        print(f"Najlepsie z testovanych: {winner[1]} (priemer {winner[0]:.5g}).")
        ax.set_title(f"{name}, {D}D")
        ax.set_xlabel("Pocet ohodnoteni ucelovej funkcie")
        ax.set_ylabel("Priemer najlepsieho najdeneho skore")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.grid(True)

    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
