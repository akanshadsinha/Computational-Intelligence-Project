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

    num_gen = len(results.global_best_per_gen)
    gens = list(range(num_gen))

    fig = sp.make_subplots(
        rows=4,
        cols=2,
        row_heights=[0.18, 0.34, 0.30, 0.18],
        specs=[
            [{"type": "table", "colspan": 2}, None],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "heatmap", "colspan": 2}, None],
            [{"type": "bar"}, {"type": "table"}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    # Summary card row
    summary_table = go.Table(
        header=dict(
            values=["<b>Problem</b>", "<b>Approach</b>", "<b>Outcome</b>"],
            fill_color="#101e37",
            font=dict(color="white", size=13),
            align="left",
            height=40,
        ),
        cells=dict(
            values=[
                [
                    "Room conflicts, professor availability, capacity constraints",
                    "Island-model GA + adaptive mutation + fitness caching",
                    f"Best fitness {results.best_fitness:.4f}<br>{len(results.migration_events)} migrations"
                ]
            ],
            fill_color="#0b1227",
            font=dict(color="#e5e8ff", size=12),
            align="left",
            height=40,
        ),
    )
    fig.add_trace(summary_table, row=1, col=1)

    # Fitness convergence chart
    for iid, best_per_gen in enumerate(results.all_islands_best_per_gen):
        padded = best_per_gen + [best_per_gen[-1]] * (num_gen - len(best_per_gen))
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=padded,
                mode="lines",
                name=f"Island {iid + 1}",
                line=dict(color=ISLAND_COLORS[iid % len(ISLAND_COLORS)], width=2),
                opacity=0.7,
                hovertemplate="Island %{fullData.name}<br>Gen %{x}: %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=gens,
            y=results.global_best_per_gen,
            mode="lines",
            name="Global Best",
            line=dict(color="#ffffff", width=4, dash="dash"),
            hovertemplate="Global best<br>Gen %{x}: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    migration_gens = sorted(set(e.generation for e in results.migration_events))
    for mg in migration_gens:
        fig.add_vline(
            x=mg,
            line_width=1,
            line_dash="dash",
            line_color="rgba(255,255,255,0.25)",
            row=2,
            col=1,
        )

    # Penalty breakdown
    pb = results.penalty_details
    penalty_names = [k.replace("_", " ").title() for k in pb.keys()]
    penalty_values = list(pb.values())
    fig.add_trace(
        go.Bar(
            x=penalty_values,
            y=penalty_names,
            orientation="h",
            marker_color=["#d85a30", "#7f77dd", "#ba7517", "#1d9e75", "#c0427a", "#378add"][: len(penalty_names)],
            text=penalty_values,
            textposition="outside",
            hovertemplate="%{y}: %{x}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Timetable heatmap
    chrom = results.best_chromosome
    grid_labels = [["" for _ in TIMES] for _ in DAYS]
    grid_z = [[0 for _ in TIMES] for _ in DAYS]
    day_map = {d: i for i, d in enumerate(DAYS)}
    time_map = {t: i for i, t in enumerate(TIMES)}

    for cid, (rid, sid) in enumerate(chrom):
        slot = get_slot(sid)
        room = get_room(rid)
        course = get_course(cid)
        di = day_map[slot.day]
        ti = time_map[slot.start]
        label = f"{course.name[:20]}{'…' if len(course.name) > 20 else ''}<br>{room.name}"
        if grid_labels[di][ti]:
            grid_labels[di][ti] += "<br>" + label
            grid_z[di][ti] += 1
        else:
            grid_labels[di][ti] = label
            grid_z[di][ti] = 1

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
            textfont={"size": 11},
            colorscale="tealrose",
            reversescale=True,
            colorbar=dict(title="Courses", ticks="outside", outlinewidth=0),
            hovertemplate="%{y} %{x}<br>%{text}<extra></extra>",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # Room usage
    room_usage = {r.name: 0 for r in ROOMS}
    for rid, _ in chrom:
        room_usage[get_room(rid).name] += 1
    fig.add_trace(
        go.Bar(
            x=list(room_usage.keys()),
            y=list(room_usage.values()),
            marker_color="#5eb5ff",
            text=list(room_usage.values()),
            textposition="outside",
            hovertemplate="%{x}: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=4,
        col=1,
    )

    # Metrics table
    final_mutation = results.mutation_rates_per_gen[-1] if results.mutation_rates_per_gen else 0.0
    metrics_table = go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>"],
            fill_color="#101e37",
            font=dict(color="white", size=12),
            align="left",
            height=36,
        ),
        cells=dict(
            values=[
                [
                    "Best Fitness",
                    "Migrations",
                    "Cache Hit Rate",
                    "Final Mutation",
                    "Generations",
                ],
                [
                    f"{results.best_fitness:.4f}",
                    str(len(results.migration_events)),
                    f"{results.cache_stats.get('hit_rate_percent', 0):.1f}%",
                    f"{final_mutation:.3f}",
                    str(num_gen),
                ],
            ],
            fill_color="#0b1227",
            font=dict(color="#e5e8ff", size=12),
            align="left",
            height=32,
        ),
    )
    fig.add_trace(metrics_table, row=4, col=2)

    fig.update_layout(
        title=dict(
            text="Island Model GA — Lecture Scheduling",
            x=0.01,
            xanchor="left",
            font=dict(size=28, color="white"),
        ),
        template="plotly_dark",
        paper_bgcolor="#060d1e",
        plot_bgcolor="#07121f",
        font=dict(color="#e5e8ff"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.08)",
        ),
        margin=dict(t=110, b=40, l=50, r=40),
        height=1080,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
        tickfont=dict(color="#d8e8ff"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
        tickfont=dict(color="#d8e8ff"),
    )

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