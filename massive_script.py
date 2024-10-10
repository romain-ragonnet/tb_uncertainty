import utils as ut
from pathlib import Path
import multiprocessing as mp
from time import time
import sys
import yaml

ANALYSIS_NAME = "full"
CONFIG_MODE = "FULL"  # "TEST" or "FULL"

parent_output_folder = Path.cwd() / "outputs"

ANALYSIS_CONFIG = {
    "TEST": {
        'opti_budget': 100,

        'mcmc_chains': 4,
        'mcmc_cores': 4,
        'mcmc_tune': 100,
        'mcmc_samples': 100,

        'full_runs_burnin': 50,
        'full_runs_samples': 100,
    },
    "FULL": {
        'opti_budget': 10000,

        'mcmc_chains': 8,
        'mcmc_cores': 8,
        'mcmc_tune': 10000,
        'mcmc_samples': 100000,

        'full_runs_burnin': 50000,
        'full_runs_samples': 50000,
    }
}

model_config = {
    "start_time": 1850,
    "end_time": 2050,
    "population": 1.e6,
    "seed": 100,   
    "intervention_time": 2025,
}

intervention_params = {
    "transmission_reduction": {
        "rel_reduction": .20
    },
    "preventive_treatment": {
        "rate": .10,
        "efficacy": .8
    },
    "faster_detection": {
        "detection_rate_mutliplier": 2.
    },
    "improved_treatment": {
        "negative_outcomes_rel_reduction": .50
    }
}

fixed_params_list = [None] + [p.name for p in ut.all_priors]


if __name__ == "__main__":
    repo_home_path = Path.home() / "repo/tb_uncertainty"

    start_time = time()

    array_task_id = int(sys.argv[2])  # specific to this particular run/analysis
    fixed_param = fixed_params_list[array_task_id - 1]

    output_root_dir = Path.home() / "sh30/users/rragonnet/outputs/tb_uncertainty"
    array_job_id = sys.argv[1]  # common to all the tasks from this array job
    analysis_output_dir = output_root_dir / f"{array_job_id}_{ANALYSIS_NAME}"
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    # create country-specific output dir
    param_analysis_output_dir = analysis_output_dir / str(fixed_param)
    param_analysis_output_dir.mkdir(exist_ok=True)

    mp.set_start_method("spawn")  # previously "forkserver"

    # Load MLE params
    mle_path = repo_home_path / "data" / "mle_params.yml"
    with open(mle_path, 'r') as file:
        mle_params = yaml.safe_load(file)

    analysis_name = 'test'
    ut.run_analysis(
        fixed_param, mle_params, ANALYSIS_CONFIG[CONFIG_MODE], model_config, intervention_params, param_analysis_output_dir,
        home_path=repo_home_path
    )

    print(f"Finished in {time() - start_time} seconds", flush=True)