import pandas as pd
from seaborn import violinplot
import matplotlib.pyplot as plt

algorithms = ["rnainverse", "inforna", "rnaredprint", "learna", "meta_learna", "meta_learna_adapt"]

all_f1_data = []
all_rnapdist_data = []

print()
import sys
sys.exit()

for algorithm in algorithms:
    print(f"Results Babura: {algorithm}")
    df_badura = pd.read_csv(f"results/results_{algorithm}_badura.csv")
    print(df_badura.mean())
    print(f"Percent correct classified: {(df_badura['f1score'] == 1).sum() * 100 / len(df_badura)}")

    print(f"Results Etherna: {algorithm}")
    df_etherna = pd.read_csv(f"results/results_{algorithm}_etherna.csv")
    print(df_etherna.mean())
    print(f"Percent correct classified: {round((df_etherna['f1score'] == 1).sum() * 100 / len(df_etherna))}")

    print()

    df_badura['algorithm'] = algorithm
    df_badura['dataset'] = 'Babura'
    df_etherna['algorithm'] = algorithm
    df_etherna['dataset'] = 'Etherna'
    
    all_f1_data.append(df_badura[['f1score', 'algorithm', 'dataset']])
    all_f1_data.append(df_etherna[['f1score', 'algorithm', 'dataset']])

    all_rnapdist_data.append(df_badura[['rnapdist', 'algorithm', 'dataset']])
    all_rnapdist_data.append(df_etherna[['rnapdist', 'algorithm', 'dataset']])


all_f1_data = pd.concat(all_f1_data)
all_rnapdist_data = pd.concat(all_rnapdist_data)


def plot_violinplot(data, y):
    plt.figure(figsize=(10, 6))
    violinplot(data=data, x="algorithm", y=y, hue="dataset", split=True)
    plt.savefig(f"../pictures/violinplot_{y}.png", dpi=300)

plot_violinplot(all_f1_data, y="f1score")
plot_violinplot(all_rnapdist_data, y="rnapdist")