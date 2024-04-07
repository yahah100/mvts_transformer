import argparse
import optuna
from optuna.samplers import RandomSampler
import yaml
import subprocess
import os 


def load_params_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        params = yaml.safe_load(file)
    return params

def build_trial_from_yaml(yaml_file, trial):
    yaml_dict = load_params_from_yaml(yaml_file)

    params = {}
    for var_name, var_value_dict in yaml_dict.items():
        var_value_dict_keys = list(var_value_dict.keys())
        if var_value_dict_keys == ["values"] and isinstance(var_value_dict["values"], list):
            
            params[var_name] = trial.suggest_categorical(var_name, var_value_dict["values"])

        elif "min" in var_value_dict_keys and "max" in var_value_dict_keys:
            min = var_value_dict["min"]
            max = var_value_dict["max"]
            if "distribution" in var_value_dict_keys:
                distribution = var_value_dict["distribution"]
            else:
                distribution = "uniform"

            if distribution == "uniform":
                params[var_name] = trial.suggest_float(var_name, min, max)
            elif distribution == "loguniform":
                params[var_name] = trial.suggest_loguniform(var_name, min, max)
            elif distribution == "discrete_uniform":
                params[var_name] = trial.suggest_discrete_uniform(var_name, min, max, q=1)
            elif distribution == "int":
                params[var_name] = trial.suggest_int(var_name, min, max)
            elif distribution == "int_uniform":
                params[var_name] = trial.suggest_int(var_name, min, max)
            elif distribution == "categorical":
                params[var_name] = trial.suggest_categorical(var_name, list(range(min, max+1)))
            else:
                raise ValueError("Distribution not supported")
        else:
            raise ValueError(f"Parameter not supported {var_name}: {var_value_dict.keys()}")
    
    return params


def create_objective(
        config_file_path: str,
        python_script: str = "your_script.py",
):
    def objective(trial):
        params = build_trial_from_yaml(config_file_path, trial)

        # Prepare the command
        command = ["python", python_script]
        for key, value in params.items():
            if isinstance(value, bool):
                if value:
                    command += [f"--{key}"]
            else:
                command += [f"--{key}", str(value)]

        trial_id = trial.number
        study_name = trial.study.study_name
        print("-"* 20)
        print(f"Study: {study_name} Trial {trial_id}")
        print(f"Time: {trial.datetime_start}")
        print(f"Comman: {command}")

        # Run the script

        result = subprocess.run(command, capture_output=True, text=True)

        # Print the standard output
        print(result.stdout)
        print(result.stderr)
        # # Call the script
        # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # # Real-time output fetching
        # while True:
        #     output = process.stdout.readline()
        #     if output == '' and process.poll() is not None:  # Check if process is done
        #         print("Process finished")
        #         break
        #     if output:
        #         print(output.strip())  # Strip trailing newline

        #     # Similarly for error output:
        #     err_output = process.stderr.readline()
        #     if err_output:
        #         print(err_output.strip())
        # result = process.communicate()[0]  

        os.makedirs(f"logs/optuna/{study_name}", exist_ok=True)
        # save results into text files
        stdout_filename = f"logs/optuna/{study_name}/trial_{trial_id}_stdout.txt"
        stderr_filename = f"logs/optuna/{study_name}/trial_{trial_id}_stderr.txt"
        # Write stdout and stderr to files
        with open(stdout_filename, 'w') as f:
            f.write(result.stdout)
        with open(stderr_filename, 'w') as f:
            f.write(result.stderr)

        return 0.0 
    return objective


def main(params):
    n_trials = params.n_trials
    hparams_file_path = params.file_path
    python_script = params.python_script

    # Your script logic using the parameters
    study = optuna.create_study(sampler=RandomSampler())
    study.optimize(func=create_objective(hparams_file_path, python_script=python_script), n_trials=n_trials)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script")
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--python_script", type=str, default="src/main.py")
    parser.add_argument("--file-path", type=str, default="sweep_files/mvtstransformer.yml")
    args = parser.parse_args()
    
    main(args)