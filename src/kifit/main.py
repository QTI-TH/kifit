import matplotlib.pyplot as plt
import numpy as np

from kifit import performfit
from kifit.loadelems import Elem

nsamples = 1
elem = Elem.get("Yb")

print(elem.means_input_params)

sample = performfit.generate_element_sample(elem=elem, nsamples=nsamples)
