import json
import numpy as np
import matplotlib.pyplot as plt

counts = {}
weight_l = {}
weight_av = {}
weight_err = {}

for N in [4, 5, 6, 8, 10, 12]:
    weight_l['{}'.format(N)] = {}
    counts['{}'.format(N)] = {}
    for B in [0.04]:
        B = round(B, 2)
        weight_l['{}'.format(N)]['{}'.format(B)] = {}
        counts['{}'.format(N)]['{}'.format(B)] = {}
        weight_av['{}_{}'.format(N, B)] = []
        weight_err['{}_{}'.format(N, B)] = []
        for s in np.arange(-2.0, 2.0, 0.5):
            s = round(s, 2)
            weight_l['{}'.format(N)]['{}'.format(B)]['{}'.format(s)] = []
            counts['{}'.format(N)]['{}'.format(B)]['{}'.format(s)] = {}
            for it in range(1, 101):
                counts['{}'.format(N)]['{}'.format(B)]['{}'.format(s)]['{}'.format(it)] = []
                adapt_test = "./compute_canada_res/adapt_test_results/N_{}_B_{}_s_{}_it_{}comp_tsp_test_results.json".format(N, B, s, it)
                with open(adapt_test, 'r') as f:
                    adapt_data = json.load(f)
                chris_test = "./chris_test_results/N_{}_s_{}_it_{}_christofides_test_results.json".format(N, s, it)
                with open(chris_test, 'r') as file:
                    chris_data = json.load(file)
                    for fq in range(N):
                        if adapt_data[fq]["fqubit_{}".format(fq)]["is_H_path"] == "Is a hamiltonian path":
                            weight_l['{}'.format(N)]['{}'.format(B)]['{}'.format(s)].append(
                                adapt_data[fq]["fqubit_{}".format(fq)]["path_weight"] / round(float(chris_data[0]), 5))

            weight_av['{}_{}'.format(N, B)].append(np.average(weight_l['{}'.format(N)]['{}'.format(B)]['{}'.format(s)]))

            weight_err['{}_{}'.format(N, B)].append(
                np.std(weight_l['{}'.format(N)]['{}'.format(B)]['{}'.format(s)]) / round(np.sqrt(len(
                    weight_l['{}'.format(N)]['{}'.format(B)]['{}'.format(s)])),
                    5))
        # print(np.average(i_counts['{}'.format(N)]['{}'.format(B)]['{}'.format(d)]))
figure, axs = plt.subplots(3, 2, figsize=(10, 8), dpi = 100)
figure.suptitle(
    "Ratio of hamiltonian path weights obtained with adapt Clifford and Christofides' algorithm.",
    weight='bold')

for i, axs in enumerate(axs.flatten()):
    N = [4, 5, 6, 8, 10, 12]


    # axis1 = axs[index]
    B = [0.04]

    x = np.arange(-2.0, 2.0, 0.5)
    y1 = weight_av['{}_{}'.format(N[i], B[0])]

    yerr1 = weight_err['{}_{}'.format(N[i], B[0])]

#    y2 = weight_av['{}_{}'.format(N[i], B[1])]

#    yerr2 = weight_err['{}_{}'.format(N[i], B[1])]

    axs.errorbar(x, y1, yerr1, marker="s", label='B = {}'.format(B[0]))

#    axs.errorbar(x, y2, yerr2, marker="s", label='B = {}'.format(B[1]))

    axs.set_title('N = {}'.format(N[i]), weight='bold')
    axs.set_ylabel('Adapt/Christo.')
    axs.set_xlabel('Graph Skewness')

    axs.set_ylim(0.7, 2.55)

    axs.set_xlim(-2.5, 2.5)
    if i == 3:
        axs.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

    #    Shrink current axis by 20%
    #    box = axis1.get_position()
    #  axis1.set_position([box.x0, box.y0, box.width, box.height * 0.9])
plt.tight_layout()
plt.show()
