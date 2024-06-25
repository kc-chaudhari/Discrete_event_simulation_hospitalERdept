from pathlib import Path
import pandas as pd
from ER_simulation import simulate, process_sim_output

args = {'num_triage_nurses': 5,
        'num_doctors': 5,
        'triage_time_mean': 5.0,
        'triage_time_sd': 1.0,
        'treatment_time_mean': 20.0,
        'treatment_time_sd': 5.0,
        'discharge_time_mean': 5.0,
        'discharge_time_sd': 1.0,
        'stoptime': 480,
        'num_reps': 15,
        'seed': 4470,
        'output_path': 'output',
        'scenario': 'baseline',
        'quiet': True}

num_reps = args['num_reps']
scenario = args['scenario']

if len(args['output_path']) > 0:
    output_dir = Path.cwd() / args['output_path']
else:
    output_dir = Path.cwd()

for i in range(1, num_reps + 1):
    simulate(args, i)

# Consolidate the patient logs and compute summary stats
patient_log_stats = process_sim_output(output_dir, scenario)
print(f"\nScenario: {scenario}")
pd.set_option("display.precision", 3)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
print(patient_log_stats['patient_log_rep_stats'])
print(patient_log_stats['patient_log_ci'])

# Create a config file based on the inputs above
config_fn = f"input/{args['scenario']}.cfg"
Path(config_fn).parent.mkdir(parents=True, exist_ok=True)

with open(config_fn, "w") as config_file:
    for key, value in args.items():
        if key != 'quiet':
            config_file.write(f"--{key} {value}\n")
        else:
            if value:
                config_file.write(f"--{key}\n")
