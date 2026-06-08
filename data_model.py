"""
data_model.py
Defines the scheduling problem: courses, rooms, timeslots, professors.
A synthetic university dataset is generated here — swap in real data as needed.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import random


@dataclass
class Professor:
    id: int
    name: str
    preferred_slots: List[int]   # indices into TIMESLOTS they prefer
    unavailable_slots: List[int] # hard unavailability


@dataclass
class Course:
    id: int
    name: str
    professor_id: int
    students: int          # enrollment count
    needs_lab: bool        # must be assigned a lab room


@dataclass
class Room:
    id: int
    name: str
    capacity: int
    is_lab: bool


@dataclass
class TimeSlot:
    id: int
    day: str
    start: str
    label: str   # e.g. "Mon 08:00"


# ── Synthetic dataset ──────────────────────────────────────────────────────────

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
TIMES = ["08:00", "10:00", "12:00", "14:00", "16:00"]

TIMESLOTS: List[TimeSlot] = [
    TimeSlot(id=i * len(TIMES) + j,
             day=DAYS[i],
             start=TIMES[j],
             label=f"{DAYS[i]} {TIMES[j]}")
    for i in range(len(DAYS))
    for j in range(len(TIMES))
]  # 25 slots total

PROFESSORS: List[Professor] = [
    Professor(id=0, name="Prof. Wagner",    preferred_slots=[0,1,2,5,6],     unavailable_slots=[20,21,22,23,24]),
    Professor(id=1, name="Prof. Müller",    preferred_slots=[5,6,7,10,11],   unavailable_slots=[0,1]),
    Professor(id=2, name="Prof. Schneider", preferred_slots=[2,3,7,8,12,13], unavailable_slots=[24]),
    Professor(id=3, name="Prof. Fischer",   preferred_slots=[10,11,15,16],   unavailable_slots=[]),
    Professor(id=4, name="Prof. Weber",     preferred_slots=[3,4,8,9,13,14], unavailable_slots=[20,21]),
]

ROOMS: List[Room] = [
    Room(id=0, name="HS 1",     capacity=120, is_lab=False),
    Room(id=1, name="HS 2",     capacity=80,  is_lab=False),
    Room(id=2, name="HS 3",     capacity=60,  is_lab=False),
    Room(id=3, name="SR A",     capacity=40,  is_lab=False),
    Room(id=4, name="SR B",     capacity=30,  is_lab=False),
    Room(id=5, name="Labor 1",  capacity=30,  is_lab=True),
    Room(id=6, name="Labor 2",  capacity=25,  is_lab=True),
]

COURSES: List[Course] = [
    Course(id=0,  name="Algorithmen & Datenstrukturen", professor_id=0, students=110, needs_lab=False),
    Course(id=1,  name="Betriebssysteme",               professor_id=1, students=75,  needs_lab=False),
    Course(id=2,  name="Datenbanken",                   professor_id=2, students=55,  needs_lab=False),
    Course(id=3,  name="Rechnernetze",                  professor_id=3, students=90,  needs_lab=False),
    Course(id=4,  name="Softwaretechnik",               professor_id=4, students=35,  needs_lab=False),
    Course(id=5,  name="Programmierung Lab",            professor_id=0, students=28,  needs_lab=True),
    Course(id=6,  name="Datenbank Lab",                 professor_id=2, students=22,  needs_lab=True),
    Course(id=7,  name="Maschinelles Lernen",           professor_id=1, students=60,  needs_lab=False),
    Course(id=8,  name="Compilerbau",                   professor_id=3, students=40,  needs_lab=False),
    Course(id=9,  name="Theoretische Informatik",       professor_id=4, students=50,  needs_lab=False),
    Course(id=10, name="Computergrafik",                professor_id=2, students=45,  needs_lab=False),
    Course(id=11, name="Embedded Systems Lab",          professor_id=3, students=20,  needs_lab=True),
]

NUM_COURSES  = len(COURSES)
NUM_ROOMS    = len(ROOMS)
NUM_SLOTS    = len(TIMESLOTS)


def get_professor(pid: int) -> Professor:
    return PROFESSORS[pid]

def get_room(rid: int) -> Room:
    return ROOMS[rid]

def get_slot(sid: int) -> TimeSlot:
    return TIMESLOTS[sid]

def get_course(cid: int) -> Course:
    return COURSES[cid]