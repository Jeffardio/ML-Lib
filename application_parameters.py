"""
APPLICATION INFORMATION
"""
n_classes = 2
prior_probability = [0.5, 0.5]
C_fn = 1
C_fp = 1
effective_prior = 0.5
eff_prior = [1 - effective_prior, effective_prior]
features_name = {
    0: "Feature 1",
    1: "Feature 2",
    2: "Feauture 3",
    3: "Feature 4",
    4: "Feature 5",
    5: "Feauture 6",
    6: "Feature 7",
    7: "Feature 8",
    8: "Feauture 9",
    9: "Feature 10",
    10: "Feature 11",
    11: "Feauture 12",
}


def set_eff_prior(prior):
    eff_prior[0] = 1- prior 
    eff_prior[1] = prior

def print_application_parameters():
    print(f"Effective prior: {effective_prior}")
