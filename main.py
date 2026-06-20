"""
main.py
Run the Island Model GA for lecture scheduling and generate the dashboard.

Usage:
  python main.py                    # default settings
  python main.py --islands 6        # more islands
  python main.py --generations 300  # longer run
  python main.py --topology star    # different migration topology
"""

import argparse
import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from island_model import IslandModel
from visualize import build_dashboard, print_schedule


def parse_args():
    p = argparse.ArgumentParser(description="Island Model GA – Lecture Scheduler")
    p.add_argument("--islands",      type=int,   default=5,     help="Number of islands (default: 5)")
    p.add_argument("--pop",          type=int,   default=40,    help="Population per island (default: 40)")
    p.add_argument("--generations",  type=int,   default=200,   help="Max generations (default: 200)")
    p.add_argument("--mutation",     type=float, default=0.12,  help="Mutation rate (default: 0.12)")
    p.add_argument("--migration",    type=int,   default=20,    help="Migration interval in generations (default: 20)")
    p.add_argument("--topology",     type=str,   default="ring",
                   choices=["ring", "star", "full"],            help="Migration topology (default: ring)")
    p.add_argument("--seed",         type=int,   default=42,    help="Random seed (default: 42)")
    p.add_argument("--output",       type=str,   default="dashboard.html", help="Output HTML path")
    p.add_argument("--no-browser",   action="store_true",       help="Don't auto-open browser")
    p.add_argument("--workers",      type=int,   default=None,   help="Number of worker processes for islands (default: auto)")
    p.add_argument("--shared-cache", action="store_true",    help="Use a Manager-shared fitness cache across workers")
    p.add_argument("--save-results", type=str,   default=None,   help="Path to save run results JSON")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  🧬 Island Model GA — Lecture Scheduling")
    print("="*60)
    print(f"  Islands:          {args.islands}")
    print(f"  Population/island:{args.pop}")
    print(f"  Generations:      {args.generations}")
    print(f"  Mutation rate:    {args.mutation}")
    print(f"  Migration every:  {args.migration} generations")
    print(f"  Topology:         {args.topology}")
    print(f"  Seed:             {args.seed}")
    print("="*60 + "\n")

    model = IslandModel(
        num_islands=args.islands,
        population_size=args.pop,
        num_generations=args.generations,
        mutation_rate=args.mutation,
        migration_interval=args.migration,
        topology=args.topology,
        workers=args.workers,
        use_shared_cache=args.shared_cache,
        seed=args.seed,
    )

    print("  Running GA...")
    t0 = time.time()
    results = model.run()
    elapsed = time.time() - t0

    print(f"\n  ✓ Done in {elapsed:.1f}s")
    print(f"  Best fitness : {results.best_fitness:.5f}")
    print(f"  Migrations   : {len(results.migration_events)}")
    
    # Fitness cache statistics
    if results.cache_stats:
        cache = results.cache_stats
        print(f"\n  💾 Fitness Cache Statistics:")
        print(f"     Cache hits:      {cache['hits']:,}")
        print(f"     Cache misses:    {cache['misses']:,}")
        print(f"     Total evals:     {cache['total']:,}")
        print(f"     Hit rate:        {cache['hit_rate_percent']:.1f}%")
        print(f"     Cache size:      {cache['cache_size']:,} entries")

    # Adaptive mutation statistics
    if results.mutation_rates_per_gen:
        rates = results.mutation_rates_per_gen
        print(f"\n  🧬 Adaptive Mutation Statistics:")
        print(f"     Initial rate:   {rates[0]:.4f}")
        print(f"     Final rate:     {rates[-1]:.4f}")
        print(f"     Min rate:       {min(rates):.4f}")
        print(f"     Max rate:       {max(rates):.4f}")
        print(f"     Avg rate:       {sum(rates)/len(rates):.4f}")
    print_schedule(results)

    print("  Building dashboard...")
    build_dashboard(results, output_path=args.output)

    if not args.no_browser:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(args.output)}")

    # Optionally save results summary to JSON
    if args.save_results:
        import json
        out = {
            "params": {
                "islands": args.islands,
                "pop": args.pop,
                "generations": args.generations,
                "mutation": args.mutation,
                "migration": args.migration,
                "topology": args.topology,
                "workers": args.workers,
                "use_shared_cache": args.shared_cache,
            },
            "summary": {
                "best_fitness": results.best_fitness,
                "migrations": len(results.migration_events),
                "cache_stats": results.cache_stats,
                "mutation_stats": {
                    "initial": results.mutation_rates_per_gen[0] if results.mutation_rates_per_gen else None,
                    "final": results.mutation_rates_per_gen[-1] if results.mutation_rates_per_gen else None,
                    "avg": (sum(results.mutation_rates_per_gen)/len(results.mutation_rates_per_gen)) if results.mutation_rates_per_gen else None,
                },
            },
        }
        with open(args.save_results, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  ✓ Results saved → {args.save_results}")


if __name__ == "__main__":
    main()