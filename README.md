# Island Model GA — Lecture Scheduling

A compact implementation of an island-model genetic algorithm (GA) for automated lecture scheduling. The project optimizes course-room-time assignments while respecting hard constraints (room capacity, lab requirements, professor availability) and soft preferences.

## Features

- Island-model GA with configurable islands, migration, and topology
- Constraint-aware schedule representation and penalty-based fitness
- Exports a human-readable best schedule to the terminal
- Interactive visualization saved as `dashboard.html` using Plotly

## Requirements

- Python 3.8+
- Python packages: `numpy`, `pandas`, `plotly`

You can install dependencies with pip:

```bash
# (recommended) from the repository root
python3 -m pip install -r requirements.txt

# or install only the required packages
python3 -m pip install numpy pandas plotly
```

To generate a `requirements.txt` from your current environment:

```bash
python3 -m pip freeze > requirements.txt
```

## Usage

Run the scheduler from the repository root:

```bash
python3 main.py
```

The script prints progress and the best schedule to the console. By default it will also build a Plotly dashboard and save it as `dashboard.html` in the working directory.

## Output

- Console: final schedule and penalty breakdown
- File: `dashboard.html` — interactive visualization of the schedule

Open `dashboard.html` in your browser to inspect results and charts.

## Project Structure

- `main.py` — entry point that configures and runs the GA
- `island_model.py` — island orchestration and migration logic
- `data_model.py` — data structures for courses, rooms, professors, slots
- `operators.py` — mutation/crossover operators and helpers
- `fitness.py` — fitness evaluation and penalty calculation
- `visualize.py` — exports Plotly dashboard and printing helpers

