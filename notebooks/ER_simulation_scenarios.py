from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from ER_simulation import simulate, process_sim_output

# Define scenarios to explore
scenarios = [
    {
        'num_triage_nurses': 5,
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
        'quiet': True
    },
    # Add more scenarios here as needed
]

# Function to run simulation for each scenario
def run_simulation(scenario):
    num_reps = scenario['num_reps']
    output_dir = Path.cwd() / scenario['output_path']
    
    # Run simulations
    for i in range(1, num_reps + 1):
        simulate(scenario, i)
    
    # Consolidate patient logs and compute summary stats
    patient_log_stats = process_sim_output(output_dir, scenario['scenario'])
    
    # Print out summary statistics
    print(f"\nScenario: {scenario['scenario']}")
    pd.set_option("display.precision", 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(patient_log_stats['patient_log_rep_stats'])
    print(patient_log_stats['patient_log_ci'])
    
    # Read patient log data from CSV
    patient_log_file = output_dir / f"consolidated_ER_patient_log_{scenario['scenario']}.csv"
    patient_log_df = pd.read_csv(patient_log_file)
    
    # Calculate and print 95th percentile of time in system
    percentile_95 = patient_log_df['time_in_system'].quantile(0.95)
    print(f"95th percentile of time in system: {percentile_95:.1f} mins")
    
    # Plot histogram of time in system
    plt.figure(figsize=(10, 6))
    plt.hist(patient_log_df['time_in_system'], bins=20, edgecolor='black')
    plt.title('Distribution of Time in System')
    plt.xlabel('Time in System (minutes)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    # Optionally save configuration file
    save_config_file(scenario)

# Function to save configuration file
def save_config_file(scenario):
    config_fn = f"input/{scenario['scenario']}.cfg"
    Path(config_fn).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_fn, "w") as config_file:
        for key, value in scenario.items():
            if key != 'quiet':
                config_file.write(f"--{key} {value}\n")
            else:
                if value:
                    config_file.write(f"--{key}\n")

# Run simulations for each scenario
for scenario in scenarios:
    run_simulation(scenario)
