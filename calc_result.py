import pandas as pd

algorithms = ["rnainverse", "inforna", "rnaredprint", "learna", "meta_learna", "meta_learna_adapt"]

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