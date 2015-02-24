from theory import *
from numerics import *
from energy_models import *


def get_models(energies, num_receptors, num_substrates):
    model_theory = ReceptorTheory(energies, num_receptors)
    model_numeric = ReceptorNumerics(energies, num_receptors, num_substrates)
    return model_theory, model_numeric