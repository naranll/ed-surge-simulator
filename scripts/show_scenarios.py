"""Print what each scenario looks like, in plain text. Standard library only (no pandas needed).

Shows, for every scenario:
  1. what the hospital knows in advance (the "info" dictionary),
  2. the EXPECTED patients per hour (the forecast, built only from that information),
  3. one example TRUE day (what actually walks in; it differs a little from the forecast).

Run from the project folder:
    py -3.12 scripts/show_scenarios.py          (example day number 0)
    py -3.12 scripts/show_scenarios.py 7        (example day number 7)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.forecaster import forecast  # noqa: E402
from src.simulation.scenarios import generate, list_scenarios, truth_seed  # noqa: E402


def main() -> None:
    day = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    names = list_scenarios()
    print("Scenarios:", ", ".join(names), f"| example true day number {day}")
    print()

    expected, actual = {}, {}
    for name in names:
        truth, info = generate(name, truth_seed(day))
        print(f"{name}: known in advance -> surge_known={info['surge_known']}, known_from={info['known_from']}, "
              f"window={info['window']}, burst_possible={info['burst_possible']}")
        expected[name] = forecast(info, 0, 24)
        counts = [0] * 24
        for patient in truth:
            counts[int(patient.t // 60) % 24] += 1
        actual[name] = counts

    print()
    print("Patients per hour. 'exp' = forecast (expected), 'day' = the example true day.")
    header = f"{'hour':>5} |" + "".join(f" {n[:8]:>8} exp {n[:8]:>8} day |" for n in names)
    print(header)
    print("-" * len(header))
    for hour in range(24):
        row = f"{hour:02d}:00 |"
        for name in names:
            row += f" {expected[name][hour]:12.1f} {actual[name][hour]:12d} |"
        print(row)
    print("-" * len(header))
    total = f"{'total':>5} |"
    for name in names:
        total += f" {sum(expected[name]):12.1f} {sum(actual[name]):12d} |"
    print(total)
    print()
    print("Heatwave: extra patients mostly 12:00-19:00. Hurricane: more patients all day.")
    print("Numbers come from src/simulation/params/scenario_parameters.json (real NHAMCS 2022 shape) "
          "plus published surge percentages.")


if __name__ == "__main__":
    main()
