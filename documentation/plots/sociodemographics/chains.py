import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import documentation.plotting as plotting

SAMPLING_RATE = 0.20

def configure(context):
    context.stage("analysis.reference.hts.chains")

    context.stage(
        "analysis.synthesis.sociodemographics.chains",
        dict(sampling_rate = SAMPLING_RATE), alias = "data"
    )

def execute(context):
    plotting.setup()

    reference = context.stage("analysis.reference.hts.chains")

    data = context.stage("data")

    # PLOT: Activity chains by sex

    marginal = ("age_range", "sex", "chain")
    df = pd.merge(data[marginal], reference[marginal].rename(columns = { "weight": "reference" }))###Selection du df correspondant à "age_range", "sex", "chain"

    df = df[df["age_range"]]###Selection des elements correspondant à l'intervalle d'age

    df_female = df[df["sex"] == "female"].sort_values(by = "reference", ascending = False).head(10)

    df_male = df[df["sex"] == "male"].sort_values(by = "reference", ascending = False).head(10)

    xport_csv_personF = df_female.to_csv(r'/home/dialloaziseoumar/AziseThesis/GenerationPopulationSynthetique/lille_metropolis/output_20p_10-05-21/df_female.csv', index=None, header=True)
    xport_csv_personM = df_male.to_csv(r'/home/dialloaziseoumar/AziseThesis/GenerationPopulationSynthetique/lille_metropolis/output_20p_10-05-21/df_male.csv', index=None, header=True)

    plt.figure(figsize = plotting.WIDE_FIGSIZE)
    hts_name = context.config("hts")

    for index, (df, title) in enumerate(zip([df_male, df_female], ["Male (15-64)", "Female (15-64)"])):
        plt.subplot(1, 2, index + 1)

        plt.bar(np.arange(10), df["reference"], width = 0.4, label = "EMD", align = "edge", linewidth = 0.5, edgecolor = "white", color = plotting.COLORS[hts_name])
        plt.bar(np.arange(10) + 0.4, df["mean"] / SAMPLING_RATE, width = 0.4, label = "Synthetic", align = "edge", linewidth = 0.5, edgecolor = "white", color = plotting.COLORS["synthetic"])

        for location, (q5, q95) in enumerate(zip(df["q5"].values, df["q95"].values)):
            location += 0.4 + 0.2
            plt.plot([location, location], [q5 / SAMPLING_RATE, q95 / SAMPLING_RATE], "k", linewidth = 1)

        plt.grid()
        plt.gca().set_axisbelow(True)
        plt.gca().xaxis.grid(alpha = 0.0)

        plt.ylim([0, 3.5e5])
        plt.ylim([0, 1.0e5])
        plt.plot([np.nan], color = "k", linewidth = 1, label = "90% Conf.")

        #plt.gca().yaxis.set_major_locator(tck.FixedLocator(np.arange(100) * 1e5))
        plt.gca().yaxis.set_major_locator(tck.FixedLocator(np.arange(5) * 1e4))
        #plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(lambda x,p: "%d" % (x * 1e-3,)))
        plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(lambda x, p: "%d" % (x * 1e-3,)))

        plt.gca().xaxis.set_major_locator(tck.FixedLocator(np.arange(10) + 0.4))
        plt.gca().xaxis.set_major_formatter(tck.FuncFormatter(lambda x,p: "\n".join(df["chain"].values[p]).upper()))

        if index == 1:
            plt.gca().yaxis.set_major_formatter(tck.FixedFormatter([""] * 1000))
            plt.gca().yaxis.get_label().set_visible(False)

        handles, labels = plt.gca().get_legend_handles_labels()
        handles = [handles[-2], handles[-1], handles[-3]]
        labels = [labels[-2], labels[-1], labels[-3]]
        plt.legend(handles = handles, labels = labels, loc = "best", title = title)

        if index == 0:
            plt.ylabel("Number of persons [x1000]")
            #plt.ylabel("Number of persons")

    plt.tight_layout()
    plt.savefig("%s/activity_chains.pdf" % context.path())
    plt.close()
