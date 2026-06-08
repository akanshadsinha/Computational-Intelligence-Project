"""
visualize.py
Interactive visualization dashboard for the Island Model GA results.
Generates an HTML dashboard with:
  1. Fitness convergence curves per island + global best
  2. Migration event markers on the convergence chart
  3. Final timetable grid with conflict heatmap
  4. Penalty breakdown bar chart
  5. Island diversity comparison
"""

import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List
from island_model import RunResults
from data_model import COURSES, ROOMS, TIMESLOTS, DAYS, TIMES, get_room, get_slot, get_course, get_professor
from fitness import Chromosome, penalty_breakdown, evaluate

# ── Colour palette per island ──────────────────────────────────────────────────
ISLAND_COLORS = ["#7F77DD", "#1D9E75", "#D85A30", "#BA7517", "#378ADD", "#C0427A"]


def build_dashboard(results: RunResults, output_path: str = "dashboard.html"):
    """Build the full interactive HTML dashboard."""

    fig = sp.make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Fitness Convergence per Island",
            "Penalty Breakdown",
            "Final Timetable Schedule",
            "Island Diversity (Final Generation)",
            "Global Best Fitness over Generations",
            "Room Utilisation",
        ],
        row_heights=[0.38, 0.38, 0.24],
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "heatmap", "colspan": 2}, None],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # ── 1. Convergence curves per island ──────────────────────────────────────
    num_gen = len(results.global_best_per_gen)
    gens = list(range(num_gen))

    for iid, best_per_gen in enumerate(results.all_islands_best_per_gen):
        padded = best_per_gen + [best_per_gen[-1]] * (num_gen - len(best_per_gen))
        fig.add_trace(
            go.Scatter(
                x=gens, y=padded,
                mode="lines",
                name=f"Island {iid + 1}",
                line=dict(color=ISLAND_COLORS[iid % len(ISLAND_COLORS)], width=1.8),
                opacity=0.75,
                legendgroup="islands",
                showlegend=True,
            ),
            row=1, col=1
        )

    # Global best line
    fig.add_trace(
        go.Scatter(
            x=gens, y=results.global_best_per_gen,
            mode="lines",
            name="Global Best",
            line=dict(color="#ffffff", width=2.5, dash="dot"),
            legendgroup="global",
        ),
        row=1, col=1
    )

    # Migration event vertical lines (as shapes)
    migration_gens = sorted(set(e.generation for e in results.migration_events))
    for mg in migration_gens:
        fig.add_vline(
            x=mg, line_width=1, line_dash="dash",
            line_color="rgba(255,255,100,0.35)", row=1, col=1
        )

    # ── 2. Penalty breakdown bar chart ────────────────────────────────────────
    pb = results.penalty_details
    penalty_names  = [k.replace("_", " ").title() for k in pb.keys()]
    penalty_values = list(pb.values())
    penalty_colors = ["#D85A30", "#7F77DD", "#BA7517", "#1D9E75", "#C0427A", "#378ADD"]

    fig.add_trace(
        go.Bar(
            x=penalty_values, y=penalty_names,
            orientation="h",
            marker_color=penalty_colors[:len(penalty_names)],
            name="Penalties",
            showlegend=False,
        ),
        row=1, col=2
    )

    # ── 3. Timetable heatmap ──────────────────────────────────────────────────
    # Build a 5-day × 5-time grid, each cell = list of courses scheduled there
    chrom = results.best_chromosome
    grid_labels = [["" for _ in range(len(TIMES))] for _ in range(len(DAYS))]
    grid_z      = [[0  for _ in range(len(TIMES))] for _ in range(len(DAYS))]

    day_map  = {d: i for i, d in enumerate(DAYS)}
    time_map = {t: i for i, t in enumerate(TIMES)}

    for cid, (rid, sid) in enumerate(chrom):
        slot   = get_slot(sid)
        room   = get_room(rid)
        course = get_course(cid)
        di = day_map[slot.day]
        ti = time_map[slot.start]
        if grid_labels[di][ti]:
            grid_labels[di][ti] += "<br>"
            grid_z[di][ti] += 1      # conflict indicator
        else:
            grid_z[di][ti] = 1
        short = course.name[:18] + ("…" if len(course.name) > 18 else "")
        grid_labels[di][ti] += f"{short}<br>({room.name})"

    # Replace empty cells
    for di in range(len(DAYS)):
        for ti in range(len(TIMES)):
            if not grid_labels[di][ti]:
                grid_labels[di][ti] = "—"

    fig.add_trace(
        go.Heatmap(
            z=grid_z,
            x=TIMES,
            y=DAYS,
            text=grid_labels,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorscale=[[0, "#1a2a3a"], [0.5, "#1D9E75"], [1.0, "#D85A30"]],
            showscale=False,
            hovertemplate="Day: %{y}<br>Time: %{x}<br>%{text}<extra></extra>",
            name="Timetable",
            showlegend=False,
        ),
        row=2, col=1
    )

    # ── 4. Island diversity (fitness distributions in final gen) ──────────────
    for iid in range(len(results.all_islands_best_per_gen)):
        last_fit = results.all_islands_best_per_gen[iid][-1]
        fig.add_trace(
            go.Bar(
                x=[f"Island {iid+1}"],
                y=[last_fit],
                marker_color=ISLAND_COLORS[iid % len(ISLAND_COLORS)],
                name=f"Island {iid+1} Final",
                showlegend=False,
            ),
            row=3, col=1
        )

    # ── 5. Global best fitness (clean version, bottom left) ───────────────────
    # Already drawn in row=1,col=1; this panel shows the same smoothed
    # We repurpose this as a "Global Best Only" clean chart
    fig.add_trace(
        go.Scatter(
            x=gens,
            y=results.global_best_per_gen,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(127,119,221,0.18)",
            line=dict(color="#7F77DD", width=2),
            name="Global best",
            showlegend=False,
        ),
        row=3, col=1
    )

    # ── 6. Room utilisation ───────────────────────────────────────────────────
    room_usage = {r.name: 0 for r in ROOMS}
    for cid, (rid, sid) in enumerate(chrom):
        room_usage[get_room(rid).name] += 1

    fig.add_trace(
        go.Bar(
            x=list(room_usage.keys()),
            y=list(room_usage.values()),
            marker_color="#378ADD",
            name="Room usage",
            showlegend=False,
        ),
        row=3, col=2
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="🧬 Island Model GA — Lecture Scheduling Dashboard",
            font=dict(size=20, color="#e8eaed"),
        ),
        paper_bgcolor="#0d1b2a",
        plot_bgcolor="#0d1b2a",
        font=dict(color="#c9d1d9", size=11),
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        height=1050,
        margin=dict(t=80, b=40, l=60, r=40),
    )

    # Axis styling
    for row in range(1, 4):
        for col in range(1, 3):
            fig.update_xaxes(
                gridcolor="rgba(255,255,255,0.07)",
                zerolinecolor="rgba(255,255,255,0.1)",
                row=row, col=col
            )
            fig.update_yaxes(
                gridcolor="rgba(255,255,255,0.07)",
                zerolinecolor="rgba(255,255,255,0.1)",
                row=row, col=col
            )

    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], row=3, col=1)
    fig.update_xaxes(title_text="Generation", row=1, col=1)
    fig.update_xaxes(title_text="Generation", row=3, col=1)
    fig.update_xaxes(title_text="Penalty count", row=1, col=2)
    fig.update_yaxes(title_text="Fitness", row=1, col=1)
    fig.update_yaxes(title_text="Fitness", row=3, col=1)

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"  ✓ Dashboard saved → {output_path}")
    return fig


def print_schedule(results: RunResults):
    """Pretty-print the best schedule to stdout."""
    chrom = results.best_chromosome
    print("\n" + "="*70)
    print(f"  BEST SCHEDULE   (fitness = {results.best_fitness:.5f})")
    print("="*70)
    print(f"  {'Course':<35} {'Room':<10} {'Slot':<14} {'Prof'}")
    print("-"*70)

    entries = []
    for cid, (rid, sid) in enumerate(chrom):
        course = get_course(cid)
        room   = get_room(rid)
        slot   = get_slot(sid)
        prof   = get_professor(course.professor_id)
        entries.append((slot.day, slot.start, course.name, room.name, prof.name))

    for day, time, name, room, prof in sorted(entries, key=lambda x: (DAYS.index(x[0]), TIMES.index(x[1]))):
        print(f"  {name:<35} {room:<10} {day} {time:<8} {prof}")

    print("="*70)
    pb = results.penalty_details
    print("\n  Penalty breakdown:")
    for k, v in pb.items():
        bar = "█" * v + "░" * max(0, 10 - v)
        print(f"    {k:<25} {bar}  {v}")
    print()