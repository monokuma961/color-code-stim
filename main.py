from color_code_stim import *
import numpy as np
import matplotlib.pyplot as plt

distance = [3]#,5,7,9,11,13,15,17]#,19,21,23,25,27,29]

T = 1
n = 1000

physical_error_rate = np.linspace(0.0146,0.017,20)
logical_error_rate = []

for D in distance:
    error_rate = []
    for p in physical_error_rate:
        colorcode = ColorCode(d=D,
                        rounds=T,
                        cnot_schedule='LLB',  # Default CNOT schedule optimized in our paper.
                        p_circuit=p)
        
        det, obs = colorcode.sample(n)

        dems = {}
        for color in ['r', 'g', 'b']:
            dem1, dem2 = colorcode.decompose_detector_error_model(color)
            dems[color] = dem1, dem2  # stim.DetectorErrorModel

        preds_obs, best_color = colorcode.decode(det, dems, verbose=True, get_color=True)

        fails = np.logical_xor(obs, preds_obs)

        P = np.sum(fails)/n
        P = P + P - P*P

        error_rate.append(P)
    logical_error_rate.append(error_rate)

# =============================================================================
# for i in range(len(distance)):
#     x = physical_error_rate
#     y = np.array(logical_error_rate[i])
#     plt.plot(x, y)
# =============================================================================
print(det)
print(dems)
