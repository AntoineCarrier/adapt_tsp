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
    for B in np.arange(0.03, 0.06, 0.01):
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
                adapt_test = "./adapt_test_results/N_{}_B_{}_s_{}_it_{}_red_tsp_test_results.json".format(N, B, s, it)
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

            weight_err['{}_{}'.format(N, B)].append(np.std(weight_l['{}'.format(N)]['{}'.format(B)]['{}'.format(s)])/len(weight_l['{}'.format(N)]['{}'.format(B)]['{}'.format(s)]))
        # print(np.average(i_counts['{}'.format(N)]['{}'.format(B)]['{}'.format(d)]))
figure, axs = plt.subplots(6)
figure.suptitle(
    'Renormalized weight of hamiltonian paths as a function of the graph density for multiple energy hierarchy '
    'parameter values (B) and multiple graph sizes (N).',
    weight='bold')

for N in [4, 5, 6, 8, 10, 12]:
    n = [4, 5, 6, 8, 10, 12]

    index = int(n.index(N))
    print(index)

    axis1 = axs[index]
    B = [0.04, 0.05]

    x = np.arange(-2.0, 2.0, 0.5)
    y1 = weight_av['{}_{}'.format(N, B[0])]

    yerr1 = weight_err['{}_{}'.format(N, B[0])]

    y2 = weight_av['{}_{}'.format(N, B[1])]

    yerr2 = weight_err['{}_{}'.format(N, B[1])]

    axis1.errorbar(x, y1, yerr1, marker="s", label='B = {}'.format(B[0]))

    axis1.errorbar(x, y2, yerr2, marker="s", label='B = {}'.format(B[1]))

    axis1.set_title('N = {}'.format(N), weight='bold')
    axis1.set_ylabel('Path weight / N')
    # axis1.set_xlabel('Graph density')


#    Shrink current axis by 20%
#    box = axis1.get_position()
#  axis1.set_position([box.x0, box.y0, box.width, box.height * 0.9])


plt.show()
