import argparse
import math
import simpy
import pandas as pd
from numpy.random import default_rng
from pathlib import Path

class EmergencyRoom:
    def __init__(self, env, num_triage_nurses, num_doctors,
                 triage_time_mean, triage_time_sd,
                 treatment_time_mean, treatment_time_sd,
                 discharge_time_mean, discharge_time_sd, rg):
        """
        Primary class that encapsulates emergency room resources.

        Parameters
        ----------
        env : simpy.Environment
        num_triage_nurses : int
        num_doctors : int
        triage_time_mean : float
        triage_time_sd : float
        treatment_time_mean : float
        treatment_time_sd : float
        discharge_time_mean : float
        discharge_time_sd : float
        rg : numpy.random.Generator
        """
        self.env = env
        self.rg = rg

        self.timestamps_list = []
        self.triage_occupancy = [(0.0, 0.0)]
        self.treatment_occupancy = [(0.0, 0.0)]
        self.discharge_occupancy = [(0.0, 0.0)]

        self.triage_nurse = simpy.Resource(env, num_triage_nurses)
        self.doctor = simpy.Resource(env, num_doctors)
        
        self.overall_occupancy = 0
        self.stage_occupancy = {'triage': 0, 'treatment': 0, 'discharge': 0}

        self.triage_time_mean = triage_time_mean
        self.triage_time_sd = triage_time_sd
        self.treatment_time_mean = treatment_time_mean
        self.treatment_time_sd = treatment_time_sd
        self.discharge_time_mean = discharge_time_mean
        self.discharge_time_sd = discharge_time_sd

    def increment_occupancy(self):
        self.overall_occupancy += 1

    def decrement_occupancy(self):
        self.overall_occupancy -= 1

    def update_stage_occupancy(self, stage, change):
        self.stage_occupancy[stage] += change

    def triage_nurse_check(self, patient):
        yield self.env.timeout(self.rg.normal(10, 2.5))

    def triage(self, patient):
        yield self.env.timeout(self.rg.normal(self.triage_time_mean, self.triage_time_sd))

    def treatment(self, patient):
        yield self.env.timeout(self.rg.normal(self.treatment_time_mean, self.treatment_time_sd))

    def discharge(self, patient):
        delay = max(0, self.rg.normal(self.discharge_time_mean, self.discharge_time_sd))
        yield self.env.timeout(delay)


class Patient:
    def __init__(self, patient_id, env):
        self.patient_id = patient_id
        self.env = env
        self.arrival_time = env.now
        self.wait_for_triage_check = 0
        self.wait_for_triage = 0
        self.wait_for_treatment = 0
        self.departure_time = None

    def get_arrival_timestamp(self):
        return self.arrival_time

    def record_triage_check_start(self):
        self.triage_check_start = self.env.now

    def record_triage_start(self):
        self.triage_start = self.env.now

    def record_treatment_start(self):
        self.treatment_start = self.env.now

    def record_departure(self):
        self.departure_time = self.env.now

    def get_departure_timestamp(self):
        return self.departure_time


def patient_arrival(env, patient_id, er):
    patient = Patient(patient_id, env)
    
    # Increment overall occupancy
    er.increment_occupancy()
    
    # Update triage stage occupancy
    er.update_stage_occupancy('triage', 1)

    # Record arrival time
    print(f"Patient {patient.patient_id} arrived at {patient.get_arrival_timestamp()}")

    # Triage Nurse Check
    with er.triage_nurse.request() as request:
        yield request
        patient.wait_for_triage_check = env.now - patient.arrival_time
        patient.record_triage_check_start()
        yield env.process(er.triage_nurse_check(patient))

    # Triage Process
    with er.triage_nurse.request() as request:
        yield request
        patient.wait_for_triage = env.now - (patient.arrival_time + patient.wait_for_triage_check)
        patient.record_triage_start()
        yield env.process(er.triage(patient))
        er.update_stage_occupancy('triage', 1)  # Update triage occupancy when starting triage

    # Treatment Process
    with er.doctor.request() as request:
        yield request
        patient.wait_for_treatment = env.now - (patient.arrival_time + patient.wait_for_triage_check + patient.wait_for_triage)
        patient.record_treatment_start()
        er.update_stage_occupancy('triage', -1)  # Update triage occupancy when leaving triage
        er.update_stage_occupancy('treatment', 1)  # Update treatment occupancy when starting treatment
        yield env.process(er.treatment(patient))  
        er.update_stage_occupancy('treatment', -1)  # Update treatment occupancy when ending treatment

    # Discharge
    with er.triage_nurse.request() as request:
        yield request
        yield env.process(er.discharge(patient))

    patient.record_departure()
    er.timestamps_list.append({
        'patient_id': patient.patient_id,
        'arrival_ts': patient.arrival_time,
        'triage_check_start_ts': patient.triage_check_start,
        'triage_start_ts': patient.triage_start,
        'treatment_start_ts': patient.treatment_start,
        'departure_ts': patient.departure_time
    })

    # Decrement overall occupancy
    er.decrement_occupancy()
    
    # Print departure time
    print(f"Patient {patient.patient_id} departed at {patient.get_departure_timestamp()}")


def run_er(env, er, stoptime=simpy.core.Infinity, max_arrivals=simpy.core.Infinity, quiet=False):
    patient_id = 0
    while env.now < stoptime and patient_id < max_arrivals:
        iat = er.rg.exponential(5)
        yield env.timeout(iat)
        patient_id += 1
        env.process(patient_arrival(env, patient_id, er))

    print(f"{patient_id} patients processed.")


def compute_durations(timestamp_df):
    timestamp_df['wait_for_triage_nurse'] = timestamp_df['triage_check_start_ts'] - timestamp_df['arrival_ts']
    timestamp_df['wait_for_triage'] = timestamp_df['triage_start_ts'] - timestamp_df['arrival_ts']
    timestamp_df['wait_for_treatment'] = timestamp_df['treatment_start_ts'] - timestamp_df['triage_start_ts']
    timestamp_df['time_in_system'] = timestamp_df['departure_ts'] - timestamp_df['arrival_ts']
    return timestamp_df


def simulate(arg_dict, rep_num):
    """
    Simulate emergency room operations based on given parameters.

    Parameters
    ----------
    arg_dict : dict
        Dictionary containing simulation parameters.
    rep_num : int
        Replication number of the simulation.

    Returns
    -------
    None
        Outputs CSV files containing simulation logs.

    """
    seed = arg_dict['seed'] + rep_num - 1
    rg = default_rng(seed=seed)
    env = simpy.Environment()

    # Initialize EmergencyRoom instance
    er = EmergencyRoom(env,
                       arg_dict['num_triage_nurses'], arg_dict['num_doctors'],
                       arg_dict['triage_time_mean'], arg_dict['triage_time_sd'],
                       arg_dict['treatment_time_mean'], arg_dict['treatment_time_sd'],
                       arg_dict['discharge_time_mean'], arg_dict['discharge_time_sd'],
                       rg)

    # Start simulation process
    env.process(run_er(env, er, stoptime=arg_dict['stoptime'], quiet=arg_dict['quiet']))
    env.run()

    # Define output directory
    if len(arg_dict['output_path']) > 0:
        output_dir = Path.cwd() / arg_dict['output_path']
    else:
        output_dir = Path.cwd()

    # Define paths for output CSV files
    er_patient_log_path = output_dir / f'er_patient_log_{arg_dict["scenario"]}_{rep_num}.csv'

    # Create patient log DataFrame and add scenario and replication number columns
    er_patient_log_df = pd.DataFrame(er.timestamps_list)
    er_patient_log_df['scenario'] = arg_dict['scenario']
    er_patient_log_df['rep_num'] = rep_num

    # Reorder columns to have scenario and rep_num first
    num_cols = len(er_patient_log_df.columns)
    new_col_order = [-2, -1] + list(range(num_cols - 2))
    er_patient_log_df = er_patient_log_df.iloc[:, new_col_order]

    # Compute durations of interest for patient log
    er_patient_log_df = compute_durations(er_patient_log_df)

    # Export patient log DataFrame to CSV
    er_patient_log_df.to_csv(er_patient_log_path, index=False)

    # Print simulation end time
    end_time = env.now
    print(f"Simulation replication {rep_num} ended at time {end_time}")


def process_sim_output(csvs_path, scenario):
    """
    Consolidates patient log CSV files, computes summary statistics, and deletes individual log files.

    Parameters
    ----------
    csvs_path : Path object or str
        Path to the directory containing simulation output patient log CSV files.
    scenario : str
        Scenario name.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'patient_log_rep_stats': DataFrame containing summary statistics for each replication.
        - 'patient_log_ci': Dictionary containing overall statistics and confidence intervals.
    """

    # Define the destination path for the consolidated CSV file
    dest_path = csvs_path / f"consolidated_ER_patient_log_{scenario}.csv"

    # Define keys to sort the DataFrame
    sort_keys = ['scenario', 'rep_num']

    # Create an empty dictionary to hold DataFrames created from each CSV file
    dfs = {}

    # Loop over all CSV files in the directory
    for csv_f in csvs_path.glob('ER_patient_log_*.csv'):
        fstem = csv_f.stem  # Get the filename stem without extension
        df = pd.read_csv(csv_f)  # Read the CSV file into a DataFrame
        dfs[fstem] = df  # Store the DataFrame in the dictionary using filename stem as key

    # Concatenate all DataFrames into one big DataFrame
    patient_log_df = pd.concat(dfs.values())

    # Sort the final DataFrame based on scenario and replication number
    patient_log_df.sort_values(sort_keys, inplace=True)

    # Export the consolidated DataFrame to a CSV file without index
    patient_log_df.to_csv(dest_path, index=False)

    # Compute summary statistics for each performance measure grouped by replication number
    patient_log_rep_stats = summarize_patient_log(patient_log_df, scenario)

    # Delete the individual replication files
    for csv_f in csvs_path.glob('ER_patient_log_*.csv'):
        csv_f.unlink()

    # Return a dictionary with summary statistics and confidence intervals
    return {
        'patient_log_rep_stats': patient_log_rep_stats,
        'patient_log_ci': {}  # Placeholder for patient_log_ci
    }


def summarize_patient_log(patient_log_df, scenario):
    """
    Summarizes patient log DataFrame by computing statistics and confidence intervals for each performance measure.

    Parameters
    ----------
    patient_log_df : pandas.DataFrame
        DataFrame containing patient log data.
    scenario : str
        Scenario name.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'wait_for_triage_nurse': Dictionary with statistics and confidence intervals for wait time for triage nurse.
        - 'wait_for_triage': Dictionary with statistics and confidence intervals for wait time for triage.
        - 'wait_for_treatment': Dictionary with statistics and confidence intervals for wait time for treatment.
        - 'time_in_system': Dictionary with statistics and confidence intervals for total time in system.
    """
    # Performance measures to summarize
    performance_measures = ['wait_for_triage_nurse', 'wait_for_triage', 'wait_for_treatment', 'time_in_system']

    # Dictionary to store results
    patient_log_stats = {}
    patient_log_rep_stats = {}  # Will store dataframes from describe on group by rep num. Keys are perf measures.
    patient_log_ci = {}        

    for pm in performance_measures:
        # Compute descriptive statistics for each replication
        rep_stats = patient_log_df.groupby(['rep_num'])[pm].describe()

        # Calculate mean across replications
        mean_mean = rep_stats['mean'].mean()
        sd_mean = rep_stats['mean'].std()
        n_samples = len(rep_stats)

        # Calculate confidence interval (assuming normal distribution for illustration)
        # Replace with appropriate method for your data if needed
        ci_95_lower = mean_mean - 1.96 * sd_mean / math.sqrt(n_samples)
        ci_95_upper = mean_mean + 1.96 * sd_mean / math.sqrt(n_samples)

        # Store statistics and confidence intervals in a dictionary
        patient_log_ci[pm] = {
            'mean': mean_mean,
            'std': sd_mean,
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper,
            'n_samples': n_samples
        }

    patient_log_stats['scenario'] = scenario
    patient_log_stats['patient_log_rep_stats'] = patient_log_rep_stats
    # Convert the final summary stats dict to a DataFrame
    patient_log_stats['patient_log_ci'] = pd.DataFrame(patient_log_ci)

    # Prepare the final dictionary to return
    return {
           'patient_log_rep_stats': patient_log_stats
    }

def process_command_line():
    parser = argparse.ArgumentParser(description='ER Simulation Parameters')
    parser.add_argument('--num_triage_nurses', type=int, default=5, help='Number of triage nurses')
    parser.add_argument('--num_doctors', type=int, default=5, help='Number of doctors')
    parser.add_argument('--triage_time_mean', type=float, default=5, help='Mean triage time')
    parser.add_argument('--triage_time_sd', type=float, default=1, help='Standard deviation of triage time')
    parser.add_argument('--treatment_time_mean', type=float, default=20, help='Mean treatment time')
    parser.add_argument('--treatment_time_sd', type=float, default=5, help='Standard deviation of treatment time')
    parser.add_argument('--discharge_time_mean', type=float, default=5, help='Mean discharge time')
    parser.add_argument('--discharge_time_sd', type=float, default=1, help='Standard deviation of discharge time')
    parser.add_argument('--stoptime', type=int, default=480, help='Total simulation time in minutes')
    parser.add_argument('--quiet', action='store_true', help='Run simulation quietly')
    parser.add_argument('--output_path', type=str, default='output', help='Path to output directory')
    parser.add_argument('--scenario', type=str, default='baseline', help='Scenario name')
    parser.add_argument('--num_reps', type=int, default=1, help='Number of repetitions for the simulation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the simulation')
    return parser.parse_args()


def main():
    args = process_command_line()
    print(args)

    num_reps = args.num_reps
    scenario = args.scenario

    if len(args.output_path) > 0:
        output_dir = Path.cwd() / args.output_path
    else:
        output_dir = Path.cwd()

    # Main simulation replication loop
    for i in range(1, num_reps + 1):
        simulate(vars(args), i)

    # Consolidate the patient logs and compute summary stats
    patient_log_stats = process_sim_output(output_dir, scenario)
    print(f"\nScenario: {scenario}")
    pd.set_option("display.precision", 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(patient_log_stats['patient_log_rep_stats'])
    print(patient_log_stats['patient_log_ci'])


if __name__ == '__main__':
    main()


