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
from fitness import Chromosome, evaluate, random_chromosome, penalty_breakdown
from operators import crossover, mutate, tournament_selection


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
        pop = self.islands[island_id]
        self.fitnesses[island_id] = [evaluate(c) for c in pop]

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

        for gen in range(self.num_generations):
            # Evolve each island
            for iid in range(self.num_islands):
                self.islands[iid] = self._evolve_island(iid)
                self._evaluate_island(iid)

            # Collect stats
            gen_best = 0.0
            for iid in range(self.num_islands):
                fits = self.fitnesses[iid]
                best_f  = max(fits)
                avg_f   = sum(fits) / len(fits)
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
                break

        return RunResults(
            best_chromosome=overall_best_chrom,
            best_fitness=overall_best_fit,
            penalty_details=penalty_breakdown(overall_best_chrom),
            island_stats=island_stats,
            migration_events=migration_events,
            global_best_per_gen=global_best_per_gen,
            all_islands_best_per_gen=all_islands_best,
        )