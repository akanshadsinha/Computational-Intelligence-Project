"""
operators.py
Domain-specific crossover and mutation operators for the scheduling GA.

Operators:
  Crossover:
    - room_swap_crossover      : swap all assignments for a random room between parents
    - professor_block_crossover: preserve one professor's full week from parent 1
    - uniform_crossover        : standard baseline (gene-by-gene coin flip)

  Mutation:
    - time_shift_mutation      : move a lecture ±1 timeslot
    - room_swap_mutation       : reassign a lecture to a different room
    - random_reset_mutation    : fully randomise one gene (fallback)
"""

import random
from typing import List
from data_model import (
    COURSES, NUM_COURSES, NUM_ROOMS, NUM_SLOTS,
    get_course, get_room
)
from fitness import Chromosome, Gene


# ── Crossover operators ────────────────────────────────────────────────────────

def room_swap_crossover(p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
    """
    Pick a random room R.
    Child 1: take parent 1 as base, but for every course assigned to room R
             in parent 2, copy that (room, slot) into child 1.
    Child 2: mirror image.
    This preserves room-block structure from both parents.
    """
    pivot_room = random.randint(0, NUM_ROOMS - 1)

    c1 = list(p1)
    c2 = list(p2)

    for i in range(NUM_COURSES):
        if p2[i][0] == pivot_room:
            c1[i] = p2[i]
        if p1[i][0] == pivot_room:
            c2[i] = p1[i]

    return c1, c2


def professor_block_crossover(p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
    """
    Pick a random professor P.
    Child 1: take parent 1 as base, but copy all of professor P's assignments
             from parent 2 (preserves their full weekly block from p2).
    Child 2: mirror.
    Keeps a professor's schedule coherent from one parent.
    """
    from data_model import PROFESSORS
    pivot_prof = random.randint(0, len(PROFESSORS) - 1)

    c1 = list(p1)
    c2 = list(p2)

    for i in range(NUM_COURSES):
        if COURSES[i].professor_id == pivot_prof:
            c1[i] = p2[i]
            c2[i] = p1[i]

    return c1, c2


def uniform_crossover(p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
    """Standard baseline: each gene independently chosen from either parent."""
    c1, c2 = [], []
    for g1, g2 in zip(p1, p2):
        if random.random() < 0.5:
            c1.append(g1); c2.append(g2)
        else:
            c1.append(g2); c2.append(g1)
    return c1, c2


def crossover(p1: Chromosome, p2: Chromosome, method: str = "mixed") -> tuple[Chromosome, Chromosome]:
    """
    Dispatcher. method='mixed' randomly picks a domain operator each call,
    which keeps diversity high across the island model.
    """
    if method == "room_swap":
        return room_swap_crossover(p1, p2)
    elif method == "prof_block":
        return professor_block_crossover(p1, p2)
    elif method == "uniform":
        return uniform_crossover(p1, p2)
    else:  # mixed — randomly choose
        choice = random.random()
        if choice < 0.4:
            return room_swap_crossover(p1, p2)
        elif choice < 0.8:
            return professor_block_crossover(p1, p2)
        else:
            return uniform_crossover(p1, p2)


# ── Mutation operators ─────────────────────────────────────────────────────────

def time_shift_mutation(chrom: Chromosome, mutation_rate: float = 0.1) -> Chromosome:
    """
    For each gene, with probability mutation_rate, shift the timeslot by ±1.
    Stays within [0, NUM_SLOTS-1]. Room stays the same.
    Models a small schedule nudge — most natural for timetabling.
    """
    mutant = list(chrom)
    for i in range(NUM_COURSES):
        if random.random() < mutation_rate:
            rid, sid = mutant[i]
            delta = random.choice([-1, 1])
            new_sid = max(0, min(NUM_SLOTS - 1, sid + delta))
            mutant[i] = (rid, new_sid)
    return mutant


def room_swap_mutation(chrom: Chromosome, mutation_rate: float = 0.1) -> Chromosome:
    """
    For each gene, with probability mutation_rate, reassign to a different room
    that respects the lab/non-lab requirement of the course.
    Timeslot stays the same.
    """
    mutant = list(chrom)
    for i in range(NUM_COURSES):
        if random.random() < mutation_rate:
            course = get_course(i)
            rid, sid = mutant[i]
            # Filter rooms that match lab requirement
            compatible = [
                r.id for r in __import__('data_model').ROOMS
                if r.is_lab == course.needs_lab and r.id != rid
            ]
            if compatible:
                new_rid = random.choice(compatible)
                mutant[i] = (new_rid, sid)
    return mutant


def random_reset_mutation(chrom: Chromosome, mutation_rate: float = 0.05) -> Chromosome:
    """Fallback: fully randomise a gene."""
    mutant = list(chrom)
    for i in range(NUM_COURSES):
        if random.random() < mutation_rate:
            mutant[i] = (random.randint(0, NUM_ROOMS - 1),
                         random.randint(0, NUM_SLOTS - 1))
    return mutant


def mutate(chrom: Chromosome, mutation_rate: float = 0.1) -> Chromosome:
    """
    Apply all three mutation operators in sequence.
    Each operates independently at its own rate.
    """
    chrom = time_shift_mutation(chrom, mutation_rate)
    chrom = room_swap_mutation(chrom, mutation_rate * 0.7)
    chrom = random_reset_mutation(chrom, mutation_rate * 0.3)
    return chrom


# ── Selection ──────────────────────────────────────────────────────────────────

def tournament_selection(population: List[Chromosome],
                         fitnesses: List[float],
                         k: int = 3) -> Chromosome:
    """Pick k random individuals and return the fittest."""
    indices = random.sample(range(len(population)), k)
    best = max(indices, key=lambda i: fitnesses[i])
    return list(population[best])