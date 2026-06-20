"""
island_model.py
Island Model Genetic Algorithm for lecture scheduling.

Architecture:
  - N islands, each a fully independent GA population
  - Every `migration_interval` generations, top-k individuals
    migrate to neighbouring islands (ring topology by default)
  - Migration events are logged for visualization

Usage:
  from island_model import IslandModel
  im = IslandModel()
  results = im.run()
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count

from fitness import Chromosome, evaluate, random_chromosome, penalty_breakdown, FitnessCache
from operators import crossover, mutate, tournament_selection


def _evolve_and_evaluate_worker(args):
    """Worker function to evolve one island for a single generation and evaluate it.

    Args (tuple): (pop, fits, pop_size, elite_size, tournament_k, crossover_rate, mutation_rate, shared_cache)

    Returns: (new_pop, new_fits, cache_stats)
    """
    (pop, fits, pop_size, elite_size, tournament_k, crossover_rate, mutation_rate, shared_cache) = args
    import random as _random
    import copy as _copy
    # Local imports from module-level to ensure picklability
    from fitness import FitnessCache
    from operators import tournament_selection, crossover as _crossover, mutate as _mutate

    # Use shared cache if provided (multiprocessing.Manager().dict())
    cache = FitnessCache(external_cache=shared_cache) if shared_cache is not None else FitnessCache()

    # Elitism
    sorted_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
    new_pop = []
    for i in range(elite_size):
        new_pop.append(_copy.deepcopy(pop[sorted_idx[i]]))

    # Fill rest
    while len(new_pop) < pop_size:
        p1 = tournament_selection(pop, fits, tournament_k)
        p2 = tournament_selection(pop, fits, tournament_k)

        if _random.random() < crossover_rate:
            c1, c2 = _crossover(p1, p2, method="mixed")
        else:
            c1, c2 = list(p1), list(p2)

        c1 = _mutate(c1, mutation_rate)
        c2 = _mutate(c2, mutation_rate)

        new_pop.append(c1)
        if len(new_pop) < pop_size:
            new_pop.append(c2)

    # Evaluate with local cache
    new_fits = [cache.evaluate(c) for c in new_pop]
    return new_pop, new_fits, cache.stats()


@dataclass
class IslandStats:
    """Per-generation stats for one island."""
    island_id: int
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float


@dataclass
class MigrationEvent:
    """Records when and which individuals migrated."""
    generation: int
    from_island: int
    to_island: int
    migrants_fitness: List[float]


@dataclass
class RunResults:
    """Everything produced by a full island model run."""
    best_chromosome: Chromosome
    best_fitness: float
    penalty_details: Dict
    island_stats: List[IslandStats]          # one entry per (island, generation)
    migration_events: List[MigrationEvent]
    global_best_per_gen: List[float]         # best fitness across all islands per generation
    all_islands_best_per_gen: List[List[float]]  # [island][generation] best fitness
    cache_stats: Dict = field(default_factory=dict)  # fitness cache hit/miss statistics
    mutation_rates_per_gen: List[float] = field(default_factory=list)  # adaptive mutation per generation


class IslandModel:
    def __init__(
        self,
        num_islands: int = 5,
        population_size: int = 40,
        num_generations: int = 200,
        mutation_rate: float = 0.12,
        crossover_rate: float = 0.85,
        migration_interval: int = 20,     # migrate every N generations
        migration_size: int = 3,          # how many individuals migrate
        topology: str = "ring",           # "ring" | "star" | "full"
        elite_size: int = 2,              # elitism: keep top N unchanged
        tournament_k: int = 4,
        adaptive_mutation: bool = True,   # enable adaptive mutation
        min_mutation_rate: float = 0.05,  # minimum mutation rate (5%)
        max_mutation_rate: float = 0.25,  # maximum mutation rate (25%)
        stagnation_patience: int = 10,    # generations without improvement before increasing mutation
        workers: Optional[int] = None,    # number of worker processes (None -> auto)
        use_shared_cache: bool = False,    # whether to use a Manager-shared cache
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.num_islands       = num_islands
        self.pop_size          = population_size
        self.num_generations   = num_generations
        self.mutation_rate     = mutation_rate
        self.crossover_rate    = crossover_rate
        self.migration_interval = migration_interval
        self.migration_size    = migration_size
        self.topology          = topology
        self.elite_size        = elite_size
        self.tournament_k      = tournament_k

        # Fitness cache for performance optimization
        self.fitness_cache     = FitnessCache()

        # Multiprocessing options
        self.workers = workers
        self.use_shared_cache = use_shared_cache
        self.shared_cache = None
        if self.use_shared_cache:
            from multiprocessing import Manager
            mgr = Manager()
            self.shared_cache = mgr.dict()

        # Adaptive mutation tracking
        self.adaptive_mutation = adaptive_mutation
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.stagnation_patience = stagnation_patience
        self.mutation_rates_per_gen: List[float] = []
        self.stagnation_counter = 0
        self.last_best_fitness = 0.0

        # Each island is a list of chromosomes
        self.islands: List[List[Chromosome]] = [
            [random_chromosome() for _ in range(population_size)]
            for _ in range(num_islands)
        ]
        self.fitnesses: List[List[float]] = [
            [0.0] * population_size for _ in range(num_islands)
        ]

    # ── Migration topology ─────────────────────────────────────────────────────

    def _migration_targets(self, island_id: int) -> List[int]:
        """Return list of islands that island_id sends migrants to."""
        n = self.num_islands
        if self.topology == "ring":
            return [(island_id + 1) % n]
        elif self.topology == "star":
            # Island 0 is hub; all others send to hub, hub sends to all
            if island_id == 0:
                return list(range(1, n))
            else:
                return [0]
        else:  # full
            return [i for i in range(n) if i != island_id]

    # ── Core GA step ───────────────────────────────────────────────────────────

    def _evaluate_island(self, island_id: int):
        """Evaluate all chromosomes on an island with fitness caching."""
        pop = self.islands[island_id]
        self.fitnesses[island_id] = [self.fitness_cache.evaluate(c) for c in pop]

    def _evolve_island(self, island_id: int) -> List[Chromosome]:
        pop  = self.islands[island_id]
        fits = self.fitnesses[island_id]
        new_pop: List[Chromosome] = []

        # Elitism: carry top individuals forward unchanged
        sorted_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
        for i in range(self.elite_size):
            new_pop.append(copy.deepcopy(pop[sorted_idx[i]]))

        # Fill rest via selection + crossover + mutation
        while len(new_pop) < self.pop_size:
            p1 = tournament_selection(pop, fits, self.tournament_k)
            p2 = tournament_selection(pop, fits, self.tournament_k)

            if random.random() < self.crossover_rate:
                c1, c2 = crossover(p1, p2, method="mixed")
            else:
                c1, c2 = list(p1), list(p2)

            c1 = mutate(c1, self.mutation_rate)
            c2 = mutate(c2, self.mutation_rate)

            new_pop.append(c1)
            if len(new_pop) < self.pop_size:
                new_pop.append(c2)

        return new_pop

    # ── Migration ──────────────────────────────────────────────────────────────

    def _migrate(self, generation: int) -> List[MigrationEvent]:
        events = []
        # Collect migrants from each island first (before any island is modified)
        migrants_per_island: List[List[Chromosome]] = []
        for iid in range(self.num_islands):
            fits = self.fitnesses[iid]
            sorted_idx = sorted(range(self.pop_size), key=lambda i: fits[i], reverse=True)
            migrants = [copy.deepcopy(self.islands[iid][i])
                        for i in sorted_idx[:self.migration_size]]
            migrants_per_island.append(migrants)

        # Send migrants to targets, replacing worst individuals on target island
        for src in range(self.num_islands):
            migrants = migrants_per_island[src]
            migrant_fits = sorted(
                [self.fitnesses[src][i] for i in range(self.pop_size)],
                reverse=True
            )[:self.migration_size]

            for dst in self._migration_targets(src):
                # Replace worst individuals on destination island
                dst_fits = self.fitnesses[dst]
                worst_idx = sorted(range(self.pop_size), key=lambda i: dst_fits[i])
                for rank, widx in enumerate(worst_idx[:self.migration_size]):
                    self.islands[dst][widx] = copy.deepcopy(migrants[rank])

                events.append(MigrationEvent(
                    generation=generation,
                    from_island=src,
                    to_island=dst,
                    migrants_fitness=migrant_fits,
                ))
        return events

    # ── Adaptive Mutation ──────────────────────────────────────────────────────

    def _update_mutation_rate(self, current_best_fit: float) -> float:
        """
        Adaptively adjust mutation rate based on fitness improvement.
        
        - If improving: reduce mutation (fine-tuning)
        - If stagnating: increase mutation (exploration)
        
        Returns the new mutation rate.
        """
        if not self.adaptive_mutation:
            return self.mutation_rate
        
        # Check for improvement
        improvement_threshold = 1e-5
        if current_best_fit > self.last_best_fitness + improvement_threshold:
            # Good progress: reduce mutation (toward exploitation)
            self.mutation_rate *= 0.95  # Reduziere um 5%
            self.stagnation_counter = 0
        else:
            # No improvement: increment stagnation counter
            self.stagnation_counter += 1
            
            # If stagnating too long: increase mutation (toward exploration)
            if self.stagnation_counter >= self.stagnation_patience:
                self.mutation_rate *= 1.15  # Erhöhe um 15%
                self.stagnation_counter = 0  # Reset counter
        
        # Clip to bounds
        self.mutation_rate = max(self.min_mutation_rate, 
                                 min(self.max_mutation_rate, self.mutation_rate))
        
        self.last_best_fitness = current_best_fit
        self.mutation_rates_per_gen.append(self.mutation_rate)
        
        return self.mutation_rate

    # ── Main run loop ──────────────────────────────────────────────────────────

    def run(self) -> RunResults:
        island_stats: List[IslandStats] = []
        migration_events: List[MigrationEvent] = []
        global_best_per_gen: List[float] = []
        all_islands_best: List[List[float]] = [[] for _ in range(self.num_islands)]

        # Initial evaluation
        for iid in range(self.num_islands):
            self._evaluate_island(iid)

        overall_best_chrom: Chromosome = copy.deepcopy(self.islands[0][0])
        overall_best_fit: float = 0.0

        # Aggregate stats for worker-local caches
        agg_hits = 0
        agg_misses = 0
        agg_cache_size = 0

        for gen in range(self.num_generations):
            # Parallel evolve+evaluate per island
            if self.num_islands > 1:
                processes = self.workers if self.workers is not None else min(self.num_islands, cpu_count())
                payloads = [(
                    self.islands[iid],
                    self.fitnesses[iid],
                    self.pop_size,
                    self.elite_size,
                    self.tournament_k,
                    self.crossover_rate,
                    self.mutation_rate,
                    self.shared_cache if self.use_shared_cache else None,
                ) for iid in range(self.num_islands)]

                with Pool(processes) as pool:
                    results = pool.map(_evolve_and_evaluate_worker, payloads)

                # Unpack results and aggregate cache stats
                for iid, (new_pop, new_fits, worker_stats) in enumerate(results):
                    self.islands[iid] = new_pop
                    self.fitnesses[iid] = new_fits
                    agg_hits += worker_stats.get("hits", 0)
                    agg_misses += worker_stats.get("misses", 0)
                    agg_cache_size += worker_stats.get("cache_size", 0)
            else:
                # Serial fallback
                for iid in range(self.num_islands):
                    self.islands[iid] = self._evolve_island(iid)
                    self._evaluate_island(iid)

            # Collect stats
            gen_best = 0.0
            for iid in range(self.num_islands):
                fits = self.fitnesses[iid]
                best_f = max(fits)
                avg_f = sum(fits) / len(fits)
                worst_f = min(fits)

                island_stats.append(IslandStats(
                    island_id=iid, generation=gen,
                    best_fitness=best_f, avg_fitness=avg_f, worst_fitness=worst_f
                ))
                all_islands_best[iid].append(best_f)

                if best_f > gen_best:
                    gen_best = best_f
                if best_f > overall_best_fit:
                    overall_best_fit = best_f
                    best_idx = fits.index(best_f)
                    overall_best_chrom = copy.deepcopy(self.islands[iid][best_idx])

            global_best_per_gen.append(gen_best)

            # ── Adaptive Mutation Update ──────────────────────────────
            self._update_mutation_rate(overall_best_fit)
            # ──────────────────────────────────────────────────────────

            # Migration step
            if (gen + 1) % self.migration_interval == 0:
                events = self._migrate(gen)
                migration_events.extend(events)

            # Early stopping
            if overall_best_fit >= 0.999:
                print(f"  ✓ Converged at generation {gen} with fitness {overall_best_fit:.5f}")
                # Pad remaining generations with last value
                remaining = self.num_generations - gen - 1
                global_best_per_gen.extend([overall_best_fit] * remaining)
                for iid in range(self.num_islands):
                    all_islands_best[iid].extend(
                        [all_islands_best[iid][-1]] * remaining
                    )

                # Pad mutation rates too
                if self.adaptive_mutation and self.mutation_rates_per_gen:
                    last_rate = self.mutation_rates_per_gen[-1]
                    self.mutation_rates_per_gen.extend([last_rate] * remaining)

                break
        # Aggregate cache stats into a single report
        total_evals = agg_hits + agg_misses
        hit_rate = (agg_hits / total_evals * 100) if total_evals > 0 else 0.0
        cache_report = {
            "hits": agg_hits,
            "misses": agg_misses,
            "total": total_evals,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": agg_cache_size,
        }

        return RunResults(
            best_chromosome=overall_best_chrom,
            best_fitness=overall_best_fit,
            penalty_details=penalty_breakdown(overall_best_chrom),
            island_stats=island_stats,
            migration_events=migration_events,
            global_best_per_gen=global_best_per_gen,
            all_islands_best_per_gen=all_islands_best,
            cache_stats=cache_report,
            mutation_rates_per_gen=self.mutation_rates_per_gen,
        )