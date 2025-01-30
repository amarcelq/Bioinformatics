#!/usr/bin/env python3
import pandas as pd
import subprocess
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from ViennaRNA import RNA
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import warnings
from typing import Iterator, Optional

warnings.simplefilter(action='ignore', category=FutureWarning)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_wall = time.time()
        start_cpu = time.process_time()
        result = func(*args, **kwargs)
        end_wall = time.time()
        end_cpu = time.process_time()
        return result, end_wall - start_wall, end_cpu - start_cpu
    return wrapper

badura_FILEPATH = "data/badura.csv"
ETERNA_FILEPATH = "data/etherna.csv"

#################
### Algorithm ###
#################
def remove_pseudoknots(structure: str) -> str:
    return ''.join(c if c in '.()' else '.' for c in structure)

def run_rnafold(sequence):
    predicted_sequence, _ = RNA.fold(sequence)
    return predicted_sequence

@measure_time
def run_rnainverse(structure: str):
    start_sequence = "".join(["N" for _ in range(len(structure))])
    predicted_sequence, _ = RNA.inverse_fold(start_sequence, structure)
    return predicted_sequence

@measure_time
def run_inforna(structure: str):
    try:
        result = subprocess.run(
            ['Algorithm/InfoRNA/InfoRNA', structure],
            text=True,
            capture_output=True, 
        )

        output_lines = result.stdout.strip().split("\n")

        if not output_lines:
            raise ValueError(f"InfoRNA returned empty output for {structure}.")

        sequence = ""
        for line in output_lines:
            if line.startswith('No valid structure!'):
                return None

            if line.startswith('Designed Sequence'):
                sequence = line.split()[-1].strip()
                break

        if not sequence.replace("N", ""):
            return None
        
        return sequence

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"InfoRNA failed with error: {e.stderr or e.stdout}"
        ) from e

    except IndexError:
        raise ValueError(
            "Unexpected output format."
        )
    
@measure_time
def run_rnaredprint(structure: str):
    result = subprocess.run(
        ['RNARedPrint', structure],
        text=True,
        capture_output=True,
    )
    output = result.stdout

    if not output:
        return ""

    sequence = output.strip().split()[1]

    return sequence

@measure_time
def run_learna(structure, 
               mutation_threshold=5, 
               batch_size=126, 
               conv_sizes=(17,5),
               conv_channels=(7,18),
               embedding_size=3,
               entropy_regularization=6.762991409135427e-05,
               fc_units=57,
               learning_rate=0.0005991629320464973,
               lstm_units=28,
               num_fc_layers=1,
               num_lstm_layers=1,
               reward_exponent=9.33503385734547,
               state_radius=32,
               restart_timeout=1800,
               timeout=600,
               restore_path=None,
               stop_learning=False):

    if not os.environ["CONDA_DEFAULT_ENV"] == "learna_tools":
        raise OSError("You first have to activate the learna_tools venv.")
    
    from Algorithm.Learna.src.learna import design_rna

    learna = design_rna.Learna()

    network_config = design_rna.NetworkConfig(
        conv_sizes=conv_sizes,  # radius * 2 + 1
        conv_channels=conv_channels,
        num_fc_layers=num_fc_layers,
        fc_units=fc_units,
        lstm_units=lstm_units,
        num_lstm_layers=num_lstm_layers,
        embedding_size=embedding_size,
    )
    agent_config = design_rna.AgentConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        entropy_regularization=entropy_regularization,
        random_agent=False,
    )
    env_config = design_rna.RnaDesignEnvironmentConfig(
        mutation_threshold=mutation_threshold,
        reward_exponent=reward_exponent,
        state_radius=state_radius,
    )

    results = learna.design_rna(
        [structure],
        timeout=timeout,
        restore_path=restore_path,
        stop_learning=stop_learning,
        restart_timeout=restart_timeout,
        network_config=network_config,
        agent_config=agent_config,
        env_config=env_config,
    )

    return results[structure]

def run_meta_learna(structure: str,
                    mutation_threshold=5,
                    batch_size=123,
                    conv_sizes=(11, 3),
                    conv_channels=(10, 3),
                    embedding_size=2,
                    entropy_regularization=0.00015087352506343337,
                    fc_units=52,
                    learning_rate=6.442010833400271e-05,
                    lstm_units=3,
                    num_fc_layers=1,
                    num_lstm_layers=0,
                    reward_exponent=8.932893783628236,
                    state_radius=29,
                    restore_path="Algorithm/Learna/models/ICLR_2019/224_0_1",
                    restart_timeout=1800,
                    stop_learning=True):

    sequence = run_learna(structure=structure,
                           mutation_threshold=mutation_threshold,
                           batch_size=batch_size,
                           conv_sizes=conv_sizes,
                           conv_channels=conv_channels,
                           embedding_size=embedding_size,
                           entropy_regularization=entropy_regularization,
                           fc_units=fc_units,
                           learning_rate=learning_rate,
                           lstm_units=lstm_units,
                           num_fc_layers=num_fc_layers,
                           num_lstm_layers=num_lstm_layers,
                           reward_exponent=reward_exponent,
                           state_radius=state_radius,
                           restore_path=restore_path,
                           restart_timeout=restart_timeout,
                           stop_learning=stop_learning)

    return sequence

def run_meta_learna_adapt(structure: str,
                          mutation_threshold=5,
                          batch_size=123,
                          conv_sizes=(11, 3),
                          conv_channels=(10, 3),
                          embedding_size=2,
                          entropy_regularization=0.00015087352506343337,
                          fc_units=52,
                          learning_rate=6.442010833400271e-05,
                          lstm_units=3,
                          num_fc_layers=1,
                          num_lstm_layers=0,
                          reward_exponent=8.932893783628236,
                          state_radius=29,
                          restore_path="Algorithm/Learna/models/ICLR_2019/224_0_1",
                          restart_timeout=1800):

    sequence = run_learna(structure=structure,
                           mutation_threshold=mutation_threshold,
                           batch_size=batch_size,
                           conv_sizes=conv_sizes,
                           conv_channels=conv_channels,
                           embedding_size=embedding_size,
                           entropy_regularization=entropy_regularization,
                           fc_units=fc_units,
                           learning_rate=learning_rate,
                           lstm_units=lstm_units,
                           num_fc_layers=num_fc_layers,
                           num_lstm_layers=num_lstm_layers,
                           reward_exponent=reward_exponent,
                           state_radius=state_radius,
                           restore_path=restore_path,
                           restart_timeout=restart_timeout)
    
    return sequence

@measure_time
def run_transformer_learna(structures):
    from TransformerLearna.main import TransformerLearna

    transformer_learna = TransformerLearna()

    if isinstance(structures, str):
        structures = [structures]

    results = transformer_learna.run(structures)

    return results

def run_algorithm(structures: list, algorithm, parallelize=True, min_parallel_size=40, sequence_by_seqence=True) -> dict:
    result: Optional(Iterator, list) = []

    if sequence_by_seqence:
        if parallelize:
            long_structures = [structure for structure in structures if len(structure) > min_parallel_size]
            short_structures = [structure for structure in structures if len(structure) <= min_parallel_size]
            
            result_short = {structure: algorithm(structure) for structure in short_structures}


            with ThreadPoolExecutor() as executor:
                result = executor.map(algorithm, long_structures)
            
            result_long = {structure: r for structure, r in zip(structures, result)}

            return {**result_short, **result_long}

        else:
            result = [algorithm(structure) for structure in structures]
            
            return {structure: r for structure, r in zip(structures, result)}
    else:
        results, wall_time, cpu_time = algorithm(structures)
        wall_time = wall_time / len(structures)
        cpu_time = cpu_time / len(structures)
        return {structure: (sequence, wall_time, cpu_time) for structure, sequence in results.items()}

###############
### Metrics ###
###############
# Idea:
# 1. Predict a sequence from structure (RNADesign Problem)
# 2. Predict a structure from the predicted sequence via rnafold
# 3. Compare them with a metric
def RNAdistance(predicted_structure: str, real_structure: str):
    """Measures the structural distance between two RNA secondary structures.
    It calculates the number of differing base pairs between the predicted and target structure.
    Measures how accurately an RNA design algorithm predicts a given target structure."""
    
    if len(predicted_structure) != len(real_structure):
        raise ValueError("Lengths of Structures have to be the same.")

    def run_rna_distance(distance: str):
        try:
            # Run RNAdistance command in a new shell
            result = subprocess.run(
                ['RNAdistance', f"--distance={distance}"],
                input=f'{predicted_structure}\n{real_structure}'.encode("utf-8"),
                stdout=subprocess.PIPE,
            )
            output_lines = result.stdout.decode('utf-8').strip().split()
            if not output_lines:
                raise ValueError(f"RNAdistance returned empty output for --distance={distance}.")
            
            # Get the last line of the output which contains the score
            score = output_lines[-1]
            return score

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"RNAdistance failed with error: {e.stderr or e.stdout}"
            ) from e

        except IndexError:
            raise ValueError(
                f"Unexpected output format from RNAdistance for --distance={distance}. Could not extract distance."
            )


    distances = ['f', 'F']

    mapping = {"f": "RNAdistance Full String Alignment", "F": "RNAdistance Full Tree Alignment"}
    result = {mapping[distance]: run_rna_distance(distance) for distance in distances}
    return result["RNAdistance Full String Alignment"], result["RNAdistance Full Tree Alignment"]
        
def RNApdist(predicted_sequence: str, real_sequence: str):
    """Compares an RNA target structure to an ensemble of possible structures derived from a sequence (using RNAfold).
    It calculates how close the target structure is to the most probable structures within the ensemble. Therefore
    good to evaluate robustness and stability.
    """
    if len(predicted_sequence) != len(real_sequence):
        raise ValueError("Lengths of Sequence have to be same.")
    
    try:
        # Run RNAdistance command in new shell
        result = subprocess.run(
            ['RNApdist'], 
            input=f'{predicted_sequence}\n{real_sequence}'.encode("utf-8"),
            stdout=subprocess.PIPE,
        )
        output_lines = result.stdout.decode('utf-8').strip().split()
        if not output_lines:
            raise ValueError("RNAdistance returned empty output.")
        
        score = output_lines[-1]

        return score

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"RNAdistance failed with error: {e.stderr or e.stdout}"
        ) from e

    except IndexError:
        raise ValueError(
            "Unexpected output format from RNAdistance. Could not extract distance."
        )

def F1Score(predicted_structure: str, real_structure: str):
    if len(predicted_structure) != len(real_structure):
        raise ValueError("Lengths of Stuctures have to be same.")

    return f1_score(list(predicted_structure), list(real_structure), average="micro")

def MCC(predicted_structure: str, real_structure: str):
    if len(predicted_structure) != len(real_structure):
        raise ValueError("Lengths of Stuctures have to be same.")

    return matthews_corrcoef(list(predicted_structure), list(real_structure))

def compute_metrics(predicted_sequence: str, real_structure: str, real_sequence: str):
    predicted_structure = run_rnafold(predicted_sequence)
    rnadist_string, rnadist_tree = RNAdistance(predicted_structure, real_structure)
    rnapdist = RNApdist(predicted_sequence, real_sequence)
    f1score = F1Score(predicted_structure, real_structure)
    mcc = MCC(predicted_structure, real_structure)

    return rnadist_string, rnadist_tree, rnapdist, f1score, mcc

def append_to_file(filename, seq_len, wall_time, cpu_time, rnadist_string, rnadist_tree, rnapdist, f1score, mcc):
    file_path = f"results/{filename}.csv"
    df = pd.DataFrame([{
        "seq_len": seq_len if seq_len else np.nan,
        "wall_time": round(wall_time, 3) if wall_time else np.nan,
        "cpu_time": round(cpu_time, 3) if cpu_time else np.nan,
        "rnadist string": rnadist_string if rnadist_string else np.nan,
        "rnadist tree": rnadist_tree if rnadist_tree else np.nan,
        "rnapdist": round(float(rnapdist), 3) if rnapdist else np.nan,
        "f1score": round(f1score, 3) if f1score else np.nan,
        "mcc": round(mcc, 3) if mcc else np.nan,
    }])
    if not os.path.exists(file_path):
        df.to_csv(file_path, mode='w', index=False, header=True)
    else:
        df.to_csv(file_path, mode='a', index=False, header=False)

def dataset_badura(frac=0.3, random_state=1):
    dataset_badura = pd.read_csv(badura_FILEPATH)[["Whole sequence", "Whole structure"]].rename(columns={"Whole sequence": "sequence", "Whole structure": "structure"})
    dataset_badura["structure"] = dataset_badura["structure"].apply(lambda x: remove_pseudoknots(x.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")))
    dataset_badura = dataset_badura.sample(frac=frac, random_state=random_state).reset_index(drop=True)

    return dataset_badura

def dataset_etherna():
    dataset_etherna = pd.read_csv(ETERNA_FILEPATH)[['Secondary Structure', "Sample Solution (1)"]].rename(columns={'Secondary Structure': "structure", "Sample Solution (1)": "sequence"})
    dataset_etherna["structure"] = dataset_etherna["structure"].apply(lambda x: remove_pseudoknots(x.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")))

    return dataset_etherna

def run(dataset_name: str, algorithm_name: str, parallelize=True, sequence_by_sequence=True):
    dataset = {"badura": dataset_badura(), "etherna": dataset_etherna()}[dataset_name]

    algorithm = {"rnainverse": run_rnainverse, 
                 "inforna": run_inforna, 
                 "rnaredprint": run_rnaredprint, 
                 "learna": run_learna, 
                 "meta_learna": run_meta_learna, 
                 "meta_learna_adapt": run_meta_learna_adapt,
                 "transformer_learna": run_transformer_learna}[algorithm_name]

    print(f"Length {dataset_name}: {len(dataset)} Algorithm: {algorithm_name}")
    chunk_size = 2
    max_structure_len = 400
    dataset = dataset[dataset["structure"].str.len() <= max_structure_len].dropna().reset_index()
    for j in range(0, len(dataset), chunk_size):
        end = j + chunk_size if j + chunk_size < len(dataset) - 1 else len(dataset) - 1
        chunk = dataset[j:end]
        print(f"{time.time()} Start Chunk\n{chunk}")
        predictions = run_algorithm(chunk["structure"].tolist(), algorithm, parallelize=parallelize, sequence_by_seqence=sequence_by_sequence)
        for i, (structure, prediction) in enumerate(predictions.items()):
            real_sequence = chunk.iloc[i]["sequence"]
            pred_sequence, wall_time, cpu_time = prediction
            if pred_sequence is not None:
                if len(pred_sequence) == len(real_sequence):
                    rnadist_string, rnadist_tree, rnapdist, f1score, mcc = compute_metrics(pred_sequence, structure, real_sequence)
                    append_to_file(f"results_{algorithm_name}_{dataset_name}", len(pred_sequence), wall_time, cpu_time, rnadist_string, rnadist_tree, rnapdist, f1score, mcc)
        print("\n")

if __name__ == "__main__":
    # run("badura", "rnainverse")
    # run("etherna", "rnainverse")

    # run("badura", "inforna")
    # run("etherna", "inforna")

    # run("badura", "rnaredprint")
    #run("etherna", "rnaredprint")

    # run("badura", "learna")
    # run("etherna", "learna")

    # run("etherna", "meta_learna")
    # run("badura", "meta_learna")

    # run("etherna", "meta_learna_adapt")
    # run("badura", "meta_learna_adapt")

    run("etherna", "transformer_learna", sequence_by_sequence=False)
    # run("badura", "transformer_learna")
