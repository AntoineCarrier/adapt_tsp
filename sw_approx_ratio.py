import json
import numpy as np
import matplotlib.pyplot as plt

counts = {}
ratios = {}
max_ratio = {}
ratio_err = {}

for N in [4]:
    ratios['{}'.format(N)] = {}
    counts['{}'.format(N)] = {}
    for B in [0.04]:
        B = round(B, 2)
        ratios['{}'.format(N)]['{}'.format(B)] = {}
        counts['{}'.format(N)]['{}'.format(B)] = {}
        max_ratio['{}_{}'.format(N, B)] = []
        ratio_err['{}_{}'.format(N, B)] = []
        for s in np.arange(-2.0, 2.0, 0.5):
            s = round(s, 2)
            ratios['{}'.format(N)]['{}'.format(B)]['{}'.format(s)] = []
            counts['{}'.format(N)]['{}'.format(B)]['{}'.format(s)] = {}
            for it in range(1, 101):
                counts['{}'.format(N)]['{}'.format(B)]['{}'.format(s)]['{}'.format(it)] = []
                adapt_test = "./compute_canada_res/adapt_test_results/N_{}_B_{}_s_{}_it_{}comp_tsp_test_results.json".format(N, B, s, it)
                with open(adapt_test, 'r') as f:
                    adapt_data = json.load(f)
                min_energy = "adapt_test_results/N_{}_standard_inst_min_ener.json".format(N)
                with open(min_energy, 'r') as file:
                    min_ener = json.load(file)
                    for fq in range(N):
                        if adapt_data[fq]["fqubit_{}".format(fq)]["is_H_path"] == "Is a hamiltonian path":
                            energy = adapt_data[fq]["fqubit_{}".format(fq)]["energy"]

                            ratio =energy/ round(float(min_ener["N_{}".format(N)]["s_{}".format(s)]["B_{}".format(B)][it]), 5)
                            if ratio > 1:
                                print(energy, min_ener["N_{}".format(N)]["s_{}".format(s)]["B_{}".format(B)][it], it, s, B)

                            ratios['{}'.format(N)]['{}'.format(B)]['{}'.format(s)].append(ratio)

            max_ratio['{}_{}'.format(N, B)].append(np.min(ratios['{}'.format(N)]['{}'.format(B)]['{}'.format(s)]))

            ratio_err['{}_{}'.format(N, B)].append(
                np.std(ratios['{}'.format(N)]['{}'.format(B)]['{}'.format(s)]) / round(np.sqrt(len(
                    ratios['{}'.format(N)]['{}'.format(B)]['{}'.format(s)])),
                    5))
        # print(np.average(i_counts['{}'.format(N)]['{}'.format(B)]['{}'.format(d)]))
figure, axs = plt.subplots(1, 1, figsize=(10, 8), dpi = 100)
figure.suptitle(
    "Energy approximation ratio of adapt-TSP.",
    weight='bold')

for i in range(1):#, axs in enumerate(axs.flatten()):
    N = [4]


    # axis1 = axs[index]
    B = [0.04]

    x = np.arange(-2.0, 2.0, 0.5)
    y1 = max_ratio['{}_{}'.format(N[i], B[0])]

    yerr1 = ratio_err['{}_{}'.format(N[i], B[0])]

    #y2 = weight_av['{}_{}'.format(N[i], B[1])]

    #yerr2 = weight_err['{}_{}'.format(N[i], B[1])]

    axs.errorbar(x, y1, yerr1, marker="s", label='B = {}'.format(B[0]))

    #axs.errorbar(x, y2, yerr2, marker="s", label='B = {}'.format(B[1]))

    axs.set_title('N = {}'.format(N[i]), weight='bold')
    axs.set_ylabel('Energy ratio')
    axs.set_xlabel('Graph Skewness')

    axs.set_ylim(0.7, 1.025)

    axs.set_xlim(-2.5, 2.5)
    if i == 3:
        axs.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

    #    Shrink current axis by 20%
    #    box = axis1.get_position()
    #  axis1.set_position([box.x0, box.y0, box.width, box.height * 0.9])
plt.tight_layout()
plt.show()
