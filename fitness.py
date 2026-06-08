"""
fitness.py
Chromosome encoding and fitness evaluation.

A chromosome is a list of (room_id, slot_id) tuples, one per course.
Index i → assignment for COURSES[i].

Fitness = 1 / (1 + total_penalty)   →  higher is better, max = 1.0
"""

import random
from typing import List, Tuple, Dict
from data_model import (
    COURSES, ROOMS, TIMESLOTS, NUM_COURSES, NUM_ROOMS, NUM_SLOTS,
    get_professor, get_room, get_slot, get_course
)

Gene = Tuple[int, int]          # (room_id, slot_id)
Chromosome = List[Gene]         # length = NUM_COURSES

# ── Penalty weights ────────────────────────────────────────────────────────────
W_ROOM_CONFLICT     = 50   # hard: two courses in same room at same time
W_PROF_CONFLICT     = 50   # hard: professor teaching two courses simultaneously
W_CAPACITY          = 30   # hard: room too small for enrollment
W_LAB_MISMATCH      = 40   # hard: lab course in non-lab room (or vice-versa)
W_PROF_UNAVAILABLE  = 35   # hard: professor scheduled when unavailable
W_PROF_PREFERENCE   = 5    # soft: professor not in preferred slot
W_STUDENT_GAP       = 2    # soft: course scheduled at edge hours (approx. gap penalty)


def random_chromosome() -> Chromosome:
    """Generate a random valid-ish chromosome."""
    return [(random.randint(0, NUM_ROOMS - 1),
             random.randint(0, NUM_SLOTS - 1))
            for _ in range(NUM_COURSES)]


def evaluate(chrom: Chromosome) -> float:
    """
    Returns fitness in [0, 1].  Higher = better schedule.
    Also returns the raw penalty for logging.
    """
    penalty = 0

    # Build lookup: slot → list of (course_idx, room_id)
    slot_room: Dict[int, List[Tuple[int, int]]] = {}
    for cid, (rid, sid) in enumerate(chrom):
        slot_room.setdefault(sid, []).append((cid, rid))

    for sid, assignments in slot_room.items():
        # Room conflict: same room used twice in the same slot
        rooms_used = [rid for _, rid in assignments]
        room_counts: Dict[int, int] = {}
        for rid in rooms_used:
            room_counts[rid] = room_counts.get(rid, 0) + 1
        for rid, cnt in room_counts.items():
            if cnt > 1:
                penalty += W_ROOM_CONFLICT * (cnt - 1)

        # Professor conflict: same professor in two slots simultaneously
        profs_used: Dict[int, int] = {}
        for cid, _ in assignments:
            pid = COURSES[cid].professor_id
            profs_used[pid] = profs_used.get(pid, 0) + 1
        for pid, cnt in profs_used.items():
            if cnt > 1:
                penalty += W_PROF_CONFLICT * (cnt - 1)

    for cid, (rid, sid) in enumerate(chrom):
        course = get_course(cid)
        room   = get_room(rid)
        slot   = get_slot(sid)
        prof   = get_professor(course.professor_id)

        # Capacity violation
        if room.capacity < course.students:
            penalty += W_CAPACITY * (course.students - room.capacity)

        # Lab mismatch
        if course.needs_lab and not room.is_lab:
            penalty += W_LAB_MISMATCH
        if not course.needs_lab and room.is_lab:
            penalty += W_LAB_MISMATCH // 2   # softer penalty for wasting a lab

        # Professor unavailability (hard)
        if sid in prof.unavailable_slots:
            penalty += W_PROF_UNAVAILABLE

        # Professor preference (soft)
        if sid not in prof.preferred_slots:
            penalty += W_PROF_PREFERENCE

        # Student gap heuristic: penalise very early (08:00) and very late (16:00) slots lightly
        if slot.start in ("08:00", "16:00"):
            penalty += W_STUDENT_GAP

    fitness = 1.0 / (1.0 + penalty)
    return fitness


def penalty_breakdown(chrom: Chromosome) -> Dict[str, int]:
    """Return named penalty components for reporting."""
    counts = {
        "room_conflicts": 0,
        "prof_conflicts": 0,
        "capacity_violations": 0,
        "lab_mismatches": 0,
        "prof_unavailable": 0,
        "soft_preference": 0,
    }

    slot_room: Dict[int, List[Tuple[int, int]]] = {}
    for cid, (rid, sid) in enumerate(chrom):
        slot_room.setdefault(sid, []).append((cid, rid))

    for sid, assignments in slot_room.items():
        rooms_used = [rid for _, rid in assignments]
        room_counts: Dict[int, int] = {}
        for rid in rooms_used:
            room_counts[rid] = room_counts.get(rid, 0) + 1
        for rid, cnt in room_counts.items():
            if cnt > 1:
                counts["room_conflicts"] += cnt - 1

        profs_used: Dict[int, int] = {}
        for cid, _ in assignments:
            pid = COURSES[cid].professor_id
            profs_used[pid] = profs_used.get(pid, 0) + 1
        for pid, cnt in profs_used.items():
            if cnt > 1:
                counts["prof_conflicts"] += cnt - 1

    for cid, (rid, sid) in enumerate(chrom):
        course = get_course(cid)
        room   = get_room(rid)
        prof   = get_professor(course.professor_id)

        if room.capacity < course.students:
            counts["capacity_violations"] += 1
        if course.needs_lab != room.is_lab:
            counts["lab_mismatches"] += 1
        if sid in prof.unavailable_slots:
            counts["prof_unavailable"] += 1
        if sid not in prof.preferred_slots:
            counts["soft_preference"] += 1

    return counts