from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput
from summer2.functions import time as stf


import yaml
from pathlib import Path   

tv_data_path = Path.cwd() / 'data' / 'time_variant_params.yml'
with open(tv_data_path, 'r') as file:
    tv_data = yaml.safe_load(file)


def prepare_intervention_processes(config: dict, intervention_params: dict, active_interventions: list):

    # Intervention-related
    if "transmission_reduction" in active_interventions:
        tv_transmission_adj = stf.get_linear_interpolation_function(
            x_pts = [config["intervention_time"], config["intervention_time"] + 1.], 
            y_pts = [1., 1. - intervention_params['transmission_reduction']['rel_reduction']]
        )
        transmission_rate = Parameter("transmission_rate") * tv_transmission_adj
    else:
        transmission_rate = Parameter("transmission_rate") 

    if "preventive_treatment" in active_interventions:
        pt_rate = stf.get_linear_interpolation_function(
            x_pts = [config["intervention_time"], config["intervention_time"] + 1.], 
            y_pts = [0., intervention_params['preventive_treatment']['rate'] * intervention_params['preventive_treatment']['efficacy']]
        )
    else:
        pt_rate = None 

    if "faster_detection" in active_interventions:
        future_detection_rates = {
            "times": [config["intervention_time"], config["intervention_time"] + 1.],
            "values": [Parameter('current_passive_detection_rate'), Parameter('current_passive_detection_rate') * intervention_params['faster_detection']["detection_rate_mutliplier"]]
        }
    else:
        future_detection_rates = {
            "times": [],
            "values": []
        }

    if "improved_treatment" in active_interventions:
        last_tsr_val = tv_data['treatment_success_perc']['values'][-1]
        future_tsr = {
            "times": [config["intervention_time"], config["intervention_time"] + 1.],
            "values": [last_tsr_val, 100. - intervention_params['improved_treatment']["negative_outcomes_rel_reduction"] * (100. - last_tsr_val)]
        }
    else:
        future_tsr = {
            "times": [],
            "values": []
        }

    return transmission_rate, pt_rate, future_detection_rates, future_tsr


def get_tb_model(config: dict, intervention_params: dict, active_interventions: list):

    """
    Prepare time-variant parameters and other quantities requiring pre-processsing
    """
    transmission_rate, pt_rate, future_detection_rates, future_tsr = prepare_intervention_processes(config, intervention_params, active_interventions)

    crude_birth_rate_func = stf.get_linear_interpolation_function(
        x_pts=tv_data['crude_birth_rate']['times'], y_pts=[cbr / 1000. for cbr in tv_data['crude_birth_rate']['values']]
    )

    life_expectancy_func = stf.get_linear_interpolation_function(
        x_pts=tv_data['life_expectancy']['times'], y_pts=tv_data['life_expectancy']['values']
    )
    all_cause_mortality_func = 1. / life_expectancy_func
    
    detection_func = stf.get_sigmoidal_interpolation_function(
        x_pts=[1950., 2024.] + future_detection_rates["times"], y_pts=[0., Parameter('current_passive_detection_rate')] + future_detection_rates["values"], curvature=16
    )

    # Treatment outcomes
    # * tx recovery rate is 1/Tx duration
    # * write equations for TSR and for prop deaths among all treatment outcomes (Pi). Solve for treatment death rate (mu_Tx) and relapse rate (phi).

    tsr_func = stf.get_linear_interpolation_function(
        x_pts=tv_data['treatment_success_perc']['times'] + future_tsr["times"], 
        y_pts=[ts_perc / 100. for ts_perc in tv_data['treatment_success_perc']['values']] + [ts_perc / 100. for ts_perc in future_tsr["values"]]
    )

    tx_recovery_rate = 1. / Parameter("tx_duration") 
    tx_death_func = tx_recovery_rate * Parameter("tx_prop_death") / tsr_func - all_cause_mortality_func
    tx_relapse_func = (all_cause_mortality_func + tx_death_func) * (1. / Parameter("tx_prop_death") - 1.) - tx_recovery_rate

    """
    Build the model
    """
    compartments = (
        "susceptible", 
        "latent_early",
        "latent_late",
        "infectious",
        "treatment", 
        "recovered",
    )
    model = CompartmentalModel(
        times=(config["start_time"], config["end_time"]),
        compartments=compartments,
        infectious_compartments=("infectious",),
    )
    model.set_initial_population(
        distribution=
        {
            "susceptible": config["population"] - config["seed"], 
            "infectious": config["seed"],
        },
    )
    
    # add birth and all cause mortality
    model.add_crude_birth_flow(
        name="birth",
        birth_rate=crude_birth_rate_func,
        dest="susceptible"
    )

    model.add_universal_death_flows(
        name="all_cause_mortality",
        death_rate= all_cause_mortality_func
    )

    # infection and reinfection flows
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=transmission_rate,
        source="susceptible", 
        dest="latent_early",
    )
    for reinfection_source in ["latent_late", "recovered"]:
        model.add_infection_frequency_flow(
            name=f"reinfection_{reinfection_source}", 
            contact_rate=transmission_rate * Parameter(f"rr_reinfection_{reinfection_source}"),
            source=reinfection_source, 
            dest="latent_early",
        )

    # latency progression
    model.add_transition_flow(
        name="stabilisation",
        fractional_rate=Parameter("stabilisation_rate"),
        source="latent_early",
        dest="latent_late",
    )
    for progression_type in ["early", "late"]:
        model.add_transition_flow(
            name=f"progression_{progression_type}",
            fractional_rate=Parameter(f"activation_rate_{progression_type}"),
            source=f"latent_{progression_type}",
            dest="infectious",
        )

    # natural recovery
    model.add_transition_flow(
        name="self_recovery",
        fractional_rate=Parameter("self_recovery_rate"),
        source="infectious",
        dest="recovered",
    )

    # TB-specific death
    model.add_death_flow(
        name="active_tb_death",
        death_rate=Parameter("tb_death_rate"),
        source="infectious",
    )

    # detection of active TB
    model.add_transition_flow(
        name="tb_detection",
        fractional_rate=detection_func,
        source="infectious",
        dest="treatment",
    )

    # treatment exit flows
    model.add_transition_flow(
        name="tx_recovery",
        fractional_rate=tx_recovery_rate,
        source="treatment",
        dest="recovered",
    )
    model.add_transition_flow(
        name="tx_relapse",
        fractional_rate=tx_relapse_func,
        source="treatment",
        dest="infectious",
    )
    model.add_death_flow(
        name="tx_death",
        death_rate=tx_death_func,
        source="treatment",
    )

    # preventive treatment
    if "preventive_treatment" in active_interventions:
        for comp in ["latent_early", "latent_late"]:       
            model.add_transition_flow(name=f"pt_{comp}", fractional_rate=pt_rate, source=comp, dest="susceptible")    


    """
       Request Derived Outputs
    """
    # Raw outputs
    model.request_output_for_compartments(name="raw_ltbi_prevalence", compartments=["latent_early", "latent_late"], save_results=False)
    model.request_output_for_compartments(name="raw_tb_prevalence", compartments=["infectious"], save_results=False)

    model.request_output_for_flow(name="progression_early", flow_name="progression_early", save_results=False)
    model.request_output_for_flow(name="progression_late", flow_name="progression_late", save_results=False)
    model.request_aggregate_output(name="raw_tb_incidence", sources=["progression_early", "progression_late"], save_results=False)

    model.request_output_for_flow(name="raw_notifications", flow_name="tb_detection")

    model.request_output_for_flow(name="active_tb_death", flow_name="active_tb_death", save_results=False)
    model.request_output_for_flow(name="tx_death", flow_name="tx_death", save_results=False)
    model.request_aggregate_output(name="all_tb_deaths", sources=["active_tb_death", "tx_death"])

    # Outputs relative to population size
    model.request_output_for_compartments(name="population", compartments=compartments)
    model.request_function_output(name="ltbi_prop", func=DerivedOutput("raw_ltbi_prevalence") / DerivedOutput("population"))
    model.request_function_output(name="tb_prevalence_per100k", func=1.e5 * DerivedOutput("raw_tb_prevalence") / DerivedOutput("population"))
    model.request_function_output(name="tb_incidence_per100k", func=1.e5 * DerivedOutput("raw_tb_incidence") / DerivedOutput("population"))
    model.request_function_output(name="tb_mortality_per100k", func=1.e5 * DerivedOutput("all_tb_deaths") / DerivedOutput("population"))

    return model
