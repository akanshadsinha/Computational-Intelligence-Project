"""
fitness.py
Chromosome encoding and fitness evaluation.

A chromosome is a list of (room_id, slot_id) tuples, one per course.
Index i → assignment for COURSES[i].

Fitness = 1 / (1 + total_penalty)   →  higher is better, max = 1.0
"""

import random
from typing import List, Tuple, Dict, Optional
from data_model import (
    COURSES, ROOMS, TIMESLOTS, NUM_COURSES, NUM_ROOMS, NUM_SLOTS,
    get_professor, get_room, get_slot, get_course
)

Gene = Tuple[int, int]          # (room_id, slot_id)
Chromosome = List[Gene]         # length = NUM_COURSES


# ── Fitness Caching ───────────────────────────────────────────────────────────

class FitnessCache:
    """
    Cache for fitness evaluations to avoid recomputing unchanged chromosomes.
    
    Usage:
      cache = FitnessCache()
      fitness = cache.evaluate(chromosome)  # cached automatically
      cache.clear()                         # reset if needed
    """
    def __init__(self, max_size: Optional[int] = None, external_cache: Optional[Dict] = None):
        """
        Args:
            max_size: Maximum cache entries. None = unlimited.
        """
        # Use external cache (e.g., multiprocessing.Manager().dict()) when provided
        if external_cache is not None:
            self.cache = external_cache
        else:
            self.cache: Dict[Tuple, float] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, chrom: Chromosome) -> Tuple:
        """Convert chromosome to hashable tuple of tuples."""
        return tuple(chrom)

    def evaluate(self, chrom: Chromosome) -> float:
        """Evaluate with caching. Returns cached value if exists."""
        key = self._key(chrom)
        
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        # Cache miss: compute and store
        self.misses += 1
        fitness = evaluate(chrom)
        
        if self.max_size is None or len(self.cache) < self.max_size:
            self.cache[key] = fitness
        
        return fitness

    def clear(self):
        """Clear all cached values."""
        self.cache.clear()

    def stats(self) -> Dict[str, int]:
        """Return cache hit/miss statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self.cache) if hasattr(self.cache, '__len__') else 0,
        }

# ── Penalty weights ────────────────────────────────────────────────────────────
W_ROOM_CONFLICT     = 50   # hard: two courses in same room at same time
W_PROF_CONFLICT     = 50   # hard: professor teaching two courses simultaneously
W_CAPACITY          = 30   # hard: room too small for enrollment
W_LAB_MISMATCH      = 40   # hard: lab course in non-lab room (or vice-versa)
W_PROF_UNAVAILABLE  = 35   # hard: professor scheduled when unavailable
W_PROF_PREFERENCE   = 5    # soft: professor not in preferred slot
W_STUDENT_GAP       = 2    # soft: course scheduled at edge hours (approx. gap penalty)


def random_chromosome() -> Chromosome:
    """Generate a random chromosome and repair obvious hard violations."""
    chrom = [(random.randint(0, NUM_ROOMS - 1),
              random.randint(0, NUM_SLOTS - 1))
             for _ in range(NUM_COURSES)]
    return repair_chromosome(chrom)


def repair_chromosome(chrom: Chromosome) -> Chromosome:
    """
    Repair obvious hard constraint violations in-place and return a new chromosome.

    Repairs performed (greedy, fast):
      - room lab / capacity mismatches (choose compatible room)
      - professor unavailability (move to preferred/available slot)
      - room conflicts (avoid double-booking where possible)
      - professor conflicts (avoid double-teaching where possible)

    This is a lightweight greedy repair to ensure initial population starts
    from a more feasible baseline.
    """
    new = list(chrom)

    # Quick helpers
    room_ids_by_lab = {
        True: [r.id for r in ROOMS if r.is_lab],
        False: [r.id for r in ROOMS if not r.is_lab]
    }

    # Track usage: slot -> set(room_ids), slot -> set(prof_ids)
    slot_room_usage = {}
    slot_prof_usage = {}
    for cid, (rid, sid) in enumerate(new):
        slot_room_usage.setdefault(sid, set()).add(rid)
        slot_prof_usage.setdefault(sid, set()).add(COURSES[cid].professor_id)

    # 1) Fix room lab/capacity mismatches
    for cid, (rid, sid) in enumerate(new):
        course = get_course(cid)
        room = get_room(rid)
        if room.capacity < course.students or course.needs_lab != room.is_lab:
            # prefer rooms that match lab requirement and fit capacity
            candidates = [r.id for r in ROOMS if r.is_lab == course.needs_lab and r.capacity >= course.students]
            if not candidates:
                # fallback: any room with correct lab flag
                candidates = [r.id for r in ROOMS if r.is_lab == course.needs_lab]
            if not candidates:
                # last resort: any room
                candidates = [r.id for r in ROOMS]

            # prefer rooms not already used in this slot
            free = [r for r in candidates if r not in slot_room_usage.get(sid, set())]
            chosen = random.choice(free) if free else random.choice(candidates)
            # update
            slot_room_usage.setdefault(sid, set()).discard(rid)
            slot_room_usage.setdefault(sid, set()).add(chosen)
            new[cid] = (chosen, sid)

    # 2) Fix professor unavailability by moving slot (and room if needed)
    for cid, (rid, sid) in enumerate(new):
        course = get_course(cid)
        prof = get_professor(course.professor_id)
        if sid in prof.unavailable_slots:
            # try preferred slots first
            candidates_slots = [s for s in prof.preferred_slots if s not in prof.unavailable_slots]
            if not candidates_slots:
                candidates_slots = [s for s in range(NUM_SLOTS) if s not in prof.unavailable_slots]

            moved = False
            for s in candidates_slots:
                # room free and professor free at s
                if new[cid][0] not in slot_room_usage.get(s, set()) and course.professor_id not in slot_prof_usage.get(s, set()):
                    # keep room if compatible, else pick compatible free room
                    rid0 = new[cid][0]
                    room0 = get_room(rid0)
                    if room0.is_lab == course.needs_lab and room0.capacity >= course.students and rid0 not in slot_room_usage.get(s, set()):
                        chosen_room = rid0
                    else:
                        compatible = [r.id for r in ROOMS if r.is_lab == course.needs_lab and r.capacity >= course.students and r.id not in slot_room_usage.get(s, set())]
                        chosen_room = random.choice(compatible) if compatible else rid0

                    # update usage
                    slot_room_usage.setdefault(s, set()).add(chosen_room)
                    slot_prof_usage.setdefault(s, set()).add(course.professor_id)
                    slot_room_usage.get(sid, set()).discard(new[cid][0])
                    slot_prof_usage.get(sid, set()).discard(course.professor_id)
                    new[cid] = (chosen_room, s)
                    moved = True
                    break

            if not moved:
                # last resort: pick any slot not in unavailable
                avail = [s for s in range(NUM_SLOTS) if s not in prof.unavailable_slots]
                if avail:
                    s = random.choice(avail)
                    compatible = [r.id for r in ROOMS if r.is_lab == course.needs_lab]
                    chosen_room = random.choice(compatible)
                    slot_room_usage.setdefault(s, set()).add(chosen_room)
                    slot_prof_usage.setdefault(s, set()).add(course.professor_id)
                    slot_room_usage.get(sid, set()).discard(new[cid][0])
                    slot_prof_usage.get(sid, set()).discard(course.professor_id)
                    new[cid] = (chosen_room, s)

    # 3) Resolve room conflicts (double-bookings)
    for sid, rooms in list(slot_room_usage.items()):
        # mapping room -> list of cids
        room_to_cids = {}
        for cid, (rid, s) in enumerate(new):
            if s == sid:
                room_to_cids.setdefault(rid, []).append(cid)

        for rid, cids in room_to_cids.items():
            if len(cids) > 1:
                # keep first, move others
                for extra_cid in cids[1:]:
                    course = get_course(extra_cid)
                    # find compatible free room at same slot
                    candidates = [r.id for r in ROOMS if r.is_lab == course.needs_lab and r.capacity >= course.students and r.id not in slot_room_usage.get(sid, set())]
                    if candidates:
                        chosen = random.choice(candidates)
                        slot_room_usage.setdefault(sid, set()).add(chosen)
                        # remove old assignment from usage
                        slot_room_usage.get(sid, set()).discard(rid)
                        new[extra_cid] = (chosen, sid)
                    else:
                        # try move to another slot where room/prof are free
                        moved = False
                        for s2 in range(NUM_SLOTS):
                            if course.professor_id in slot_prof_usage.get(s2, set()):
                                continue
                            candidates2 = [r.id for r in ROOMS if r.is_lab == course.needs_lab and r.capacity >= course.students and r.id not in slot_room_usage.get(s2, set())]
                            if candidates2:
                                chosen = random.choice(candidates2)
                                slot_room_usage.setdefault(s2, set()).add(chosen)
                                slot_prof_usage.setdefault(s2, set()).add(course.professor_id)
                                slot_room_usage.get(sid, set()).discard(rid)
                                slot_prof_usage.get(sid, set()).discard(course.professor_id)
                                new[extra_cid] = (chosen, s2)
                                moved = True
                                break
                        if not moved:
                            # fallback random
                            new[extra_cid] = (random.randint(0, NUM_ROOMS - 1), random.randint(0, NUM_SLOTS - 1))

    # 4) Resolve professor conflicts
    for s in range(NUM_SLOTS):
        prof_to_cids = {}
        for cid, (rid, sid) in enumerate(new):
            if sid == s:
                pid = COURSES[cid].professor_id
                prof_to_cids.setdefault(pid, []).append(cid)

        for pid, cids in prof_to_cids.items():
            if len(cids) > 1:
                for extra_cid in cids[1:]:
                    course = get_course(extra_cid)
                    prof = get_professor(course.professor_id)
                    found = False
                    for s2 in range(NUM_SLOTS):
                        if s2 in prof.unavailable_slots:
                            continue
                        candidates = [r.id for r in ROOMS if r.is_lab == course.needs_lab and r.capacity >= course.students and r.id not in slot_room_usage.get(s2, set())]
                        if candidates and course.professor_id not in slot_prof_usage.get(s2, set()):
                            chosen = random.choice(candidates)
                            slot_room_usage.setdefault(s2, set()).add(chosen)
                            slot_prof_usage.setdefault(s2, set()).add(course.professor_id)
                            slot_room_usage.get(s, set()).discard(new[extra_cid][0])
                            slot_prof_usage.get(s, set()).discard(course.professor_id)
                            new[extra_cid] = (chosen, s2)
                            found = True
                            break
                    if not found:
                        # fallback: choose any available slot
                        avail = [ss for ss in range(NUM_SLOTS) if ss not in prof.unavailable_slots]
                        if avail:
                            s2 = random.choice(avail)
                        else:
                            s2 = random.randint(0, NUM_SLOTS - 1)
                        candidates = [r.id for r in ROOMS if r.is_lab == course.needs_lab]
                        chosen = random.choice(candidates)
                        new[extra_cid] = (chosen, s2)

    return new


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