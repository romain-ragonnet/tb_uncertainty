import pymc as pm
import pandas as pd
import arviz as az
from copy import deepcopy

from estival.model import BayesianCompartmentalModel
from estival.wrappers import pymc as epm
import estival.priors as esp
import estival.targets as est
from estival.sampling import tools as esamptools
from estival.wrappers.nevergrad import optimize_model

import model as md

all_priors = [
    esp.UniformPrior("transmission_rate", [0.1, 10.]),
    esp.UniformPrior("activation_rate_early", [0.30681, 0.547875]),  # Epidemics 2017, converted 95CIs from daily to annual rates
    esp.UniformPrior("activation_rate_late", [0.00091, 0.00401775]), # Epidemics 2017, converted 95CIs from daily to annual rates
    esp.UniformPrior("stabilisation_rate", [3.104625, 5.1135]), # Epidemics 2017, converted 95CIs from daily to annual rates
    esp.UniformPrior("rr_reinfection_latent_late", [0.2, 0.5]),
    esp.UniformPrior("rr_reinfection_recovered", [0.5, 1.]),
    esp.UniformPrior("self_recovery_rate", [0.1, 0.3]),
    esp.UniformPrior("tb_death_rate", [0.02, 0.4]),
    esp.UniformPrior("current_passive_detection_rate", [0.2, 2.]),
]

target_data = pd.Series({2024: 1000.})

targets = [
    est.NormalTarget("tb_prevalence_per100k", target_data , stdev=100.)
]


default_params = {
    # Planning to vary these parameters
    'transmission_rate': 1.,

    'activation_rate_early': 1.,
    'activation_rate_late': 1.,
    'stabilisation_rate': 1.,

    'rr_reinfection_latent_late': 1.,
    'rr_reinfection_recovered': 1.,

    'self_recovery_rate': 1.,
    'tb_death_rate': 1.,

    'current_passive_detection_rate': 1.,

    # Planning to fix these ones
    'tx_duration': 0.5,
    'tx_prop_death': .04  # WHO, among new treatment
}

intervention_names = {
    "transmission_reduction": "transmission reduction",
    "preventive_treatment": "preventive treatment",
    "faster_detection": "faster detection",
    "improved_treatment": "improved treatment"
}


def get_bcm_object(model, params, fixed_param_name=None):
    
    priors = [p for p in all_priors if p.name != fixed_param_name]

    bcm = BayesianCompartmentalModel(model, default_params | params, priors, targets)

    return bcm


def find_mle(model, opti_budget=1000):

    bcm = get_bcm_object(model, {})
    orunner = optimize_model(bcm)
    rec = orunner.minimize(opti_budget)
    mle_params = rec.value[1]

    return mle_params


def run_sampling(model, all_mle_params, fixed_param_name, draws=10000, tune=1000, cores=4, chains=4):

    bcm = get_bcm_object(model, default_params | all_mle_params, fixed_param_name)
    with pm.Model() as model:
        variables = epm.use_model(bcm)
        idata = pm.sample(
            step=[pm.DEMetropolisZ(variables)], draws=draws, tune=tune, cores=cores, chains=chains,
            initvals={key: val for key, val in all_mle_params.items() if key != fixed_param_name}
        ) 
    
    return idata


def check_sampling(bcm, idata, n_samples=1000):
    az.plot_trace(idata)

    sample_idata = az.extract(idata, num_samples = n_samples)
    mres = esamptools.model_results_for_samples(sample_idata, bcm)
    esamptools.quantiles_for_results(mres.results,[0.025,0.25,0.5,0.75,0.975])["tb_prevalence_per100k"].loc[2010:].plot()
    target_data.loc[2010:].plot(style='.',color='red') 


def calculate_diff_output_quantiles(ref_full_runs, sc_full_runs, quantiles=[.025, .25, .5, .75, .975]):
    diff_names = {
        "deaths_averted": "cumulative_TB_deaths",
        "TB_episodes_averted": "cumulative_incidence",
    }
    
    latest_time = ref_full_runs.results.index.max()
    
    runs_0_latest = ref_full_runs.results.loc[latest_time]
    runs_1_latest = sc_full_runs.results.loc[latest_time]

    abs_diff = runs_1_latest - runs_0_latest
    rel_diff = (runs_1_latest - runs_0_latest) / runs_0_latest
    
    diff_quantiles_df_abs = pd.DataFrame(
        index=quantiles, 
        data={colname: abs_diff[output_name].quantile(quantiles) for colname, output_name in diff_names.items()}
    )   
    diff_quantiles_df_rel = pd.DataFrame(
        index=quantiles, 
        data={f"{colname}_relative" : rel_diff[output_name].quantile(quantiles) for colname, output_name in diff_names.items()}
    ) 
    
    return pd.concat([diff_quantiles_df_abs, diff_quantiles_df_rel], axis=1)




def run_analysis(fixed_param, mle_params, analysis_config, model_config, intervention_params, output_folder_path, home_path):
    
    output_folder_path.mkdir(parents=True, exist_ok=True)
    model = md.get_tb_model(model_config, intervention_params, active_interventions=[], home_path=home_path)

    if mle_params is None:
        mle_params = find_mle(model, opti_budget=analysis_config['opti_budget'])

    print(f"Running Metropolis sampling fixing {fixed_param}")
    idata = run_sampling(model, mle_params, fixed_param, draws=analysis_config['mcmc_samples'], tune=analysis_config['mcmc_tune'], cores=analysis_config['mcmc_cores'], chains=analysis_config['mcmc_chains'])
    idata.to_netcdf(output_folder_path / f"idata_{fixed_param}.nc")

    print(f"Running full runs fixing {fixed_param}")
    chain_length = idata.sample_stats.sizes['draw']
    burnt_idata = idata.sel(draw=range(analysis_config['full_runs_burnin'], chain_length))  # Discard burn-in
    full_run_param_samples =  az.extract(burnt_idata, num_samples=analysis_config['full_runs_samples'])
    (output_folder_path / "full_runs").mkdir(exist_ok=True)
    (output_folder_path / "diff_output_dfs").mkdir(exist_ok=True)
    for intervention in [None] + list(intervention_params.keys()):
        active_interventions = [intervention] if intervention else []
        model = md.get_tb_model(model_config, intervention_params, active_interventions, home_path=home_path)
        bcm = get_bcm_object(model, default_params | mle_params, fixed_param)
        full_runs = esamptools.model_results_for_samples(full_run_param_samples, bcm)
        full_runs.results.to_hdf(output_folder_path / "full_runs" / f"fullruns_{fixed_param}_{intervention}.h5", key="my_data")

        if intervention is None:
            ref_full_runs = deepcopy(full_runs)
        else:
            diff_output_dfs = calculate_diff_output_quantiles(ref_full_runs, full_runs)
            diff_output_dfs.to_csv(output_folder_path / "diff_output_dfs" / f"diff_outputs_{fixed_param}_{intervention}.csv")