import os
import numpy as np

import matplotlib.pyplot as plt
import adjustText as adjust_text

results = {
    "Positional CLEVR": {
        "StyleGAN2-ADA": {"Accuracy": [[37.28, 1.37]], "FID": [[57.41]]},
        "StyleGAN2": {
            "Accuracy": [[1.04, 0.29], [0.04, 0.04], [0.0, 0.0]],
            "FID": [[51.37], [23.29], [19.01]],
        },
        "LACE": {
            "Accuracy": [[0.7, 0.24], [0.0, 0.0], [0.0, 0.0]],
            "FID": [[50.92], [22.83], [19.62]],
        },
        "GLIDE": {
            "Accuracy": [[0.86, 0.26], [0.06, 0.06], [0.0, 0.0]],
            "FID": [[61.68], [38.26], [37.18]],
        },
        "EBM": {
            "Accuracy": [[70.54, 1.29], [28.22, 1.27], [7.34, 0.74]],
            "FID": [[78.63], [65.45], [58.33]],
        },
        "Composed GLIDE": {
            "Accuracy": [[86.42, 0.97], [59.2, 1.39], [31.36, 1.31]],
            "FID": [[29.29], [15.94], [10.51]],
        },
        "Ours": {
            "Accuracy": [[99.3, 0.24], [98.18, 0.38], [95.04, 0.61]],
            "FID": [[13.76], [15.3], [16.23]],
        },
    },
    "Relational CLEVR": {
        "StyleGAN2-ADA": {"Accuracy": [[67.71, 1.32]], "FID": [[20.55]]},
        "StyleGAN2": {
            "Accuracy": [[20.18, 1.14], [1.66, 0.36], [0.16, 0.11]],
            "FID": [[22.29], [30.58], [31.3]],
        },
        "LACE": {
            "Accuracy": [[1.1, 0.3], [0.1, 0.09], [0.04, 0.04]],
            "FID": [[40.54], [40.61], [40.6]],
        },
        "GLIDE": {
            "Accuracy": [[46.2, 1.41], [8.86, 0.8], [1.36, 0.33]],
            "FID": [[17.61], [28.56], [40.02]],
        },
        "EBM": {
            "Accuracy": [[78.14, 1.17], [24.16, 1.21], [4.26, 0.57]],
            "FID": [[44.41], [55.89], [58.66]],
        },
        "Composed GLIDE": {
            "Accuracy": [[60.4, 1.38], [21.84, 1.17], [2.8, 0.47]],
            "FID": [[29.06], [29.82], [26.11]],
        },
        "Ours": {
            "Accuracy": [[78.16, 1.17], [43.06, 1.4], [14.3, 0.99]],
            "FID": [[30.0], [28.87], [30.34]],
        },
    },
    "FFHQ Attributes": {
        "StyleGAN2-ADA": {"Accuracy": [[91.06, 0.81]], "FID": [[10.75]]},
        "StyleGAN2": {
            "Accuracy": [[58.9, 1.39], [30.68, 1.3], [16.96, 1.06]],
            "FID": [[18.04], [18.06], [18.06]],
        },
        "LACE": {
            "Accuracy": [[97.6, 0.43], [95.66, 0.58], [80.88, 1.11]],
            "FID": [[28.21], [36.23], [34.64]],
        },
        "GLIDE": {
            "Accuracy": [[98.66, 0.33], [48.68, 1.41], [27.24, 1.26]],
            "FID": [[20.3], [22.69], [21.98]],
        },
        "EBM": {
            "Accuracy": [[98.74, 0.32], [93.1, 0.72], [30.01, 1.3]],
            "FID": [[89.95], [99.64], [335.7]],
        },
        "Composed GLIDE": {
            "Accuracy": [[99.26, 0.24], [92.68, 0.74], [68.86, 1.31]],
            "FID": [[18.72], [17.22], [16.95]],
        },
        "Ours": {
            "Accuracy": [[99.78, 0.13], [99.38, 0.22], [99.18, 0.26]],
            "FID": [[21.52], [28.25], [33.8]],
        },
    },
}
np.set_printoptions(suppress=True)
plt.rcParams["axes.formatter.useoffset"] = False

# all round bigger text
plt.rcParams.update({"font.size": 14})

# but titles are smaller
plt.rcParams.update({"axes.titlesize": 16})

for dataset, dataset_results in results.items():
    for n in range(1, 4):
        data = []
        labels = []
        for method, method_results in dataset_results.items():
            try:
                accuracy = method_results["Accuracy"][n - 1][0]
                error = 100 - accuracy
                fid = method_results["FID"][n - 1][0]

                data.append((error, fid))
                labels.append(method)
            except IndexError:
                pass

        # now compute the pareto front
        pareto = []

        # add virtual point that connects the min fid point to the right of the axis
        pareto.append((max([x[0] for x in data]), min([x[1] for x in data]), ""))

        for i, (error, fid) in enumerate(data):
            if all(error2 >= error or fid2 >= fid for error2, fid2 in data):
                pareto.append((error, fid, labels[i]))

        # add virtual point that connects the min error point to the top of the axis
        pareto.append((min([x[0] for x in data]), max([x[1] for x in data]), ""))

        # sort points by error rate (if equal, sort by negative fid score)
        pareto = sorted(pareto, key=lambda x: (x[0], -x[1]))

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(*zip(*data), marker="x", color="black")

        # dotted line for pareto front
        for i in range(len(pareto) - 1):
            ax.plot(
                [pareto[i][0], pareto[i + 1][0]],
                [pareto[i][1], pareto[i + 1][1]],
                linestyle="--",
                color="black",
            )

        ax.set_xlabel("Error Rate (%)")
        ax.set_ylabel("FID")

        ax.set_title(f'{dataset} - {n} Component{"s" if n > 1 else ""}')
        anns = []
        for label, x, y in zip(labels, [x[0] for x in data], [x[1] for x in data]):
            # if ours, bold
            if label == "Ours":
                anns.append(ax.text(x, y, label, weight="bold"))
            else:
                anns.append(ax.text(x, y, label))
        adjust_text.adjust_text(
            anns, arrowprops=dict(arrowstyle="->", color="black", lw=1)
        )

        plt.tight_layout()

        if not os.path.exists("./plots"):
            os.makedirs("./plots")

        plt.savefig(f"./plots/pareto_{dataset}_{n}.pdf", bbox_inches="tight")
