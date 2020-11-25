from dendroscope_wrapper import *
from netrax_wrapper import *
from raxml_wrapper import infer_raxml_tree

from experiment_model import *

def evaluate_dataset(dataset):
    res = Result(dataset)
    _, res.bic_true, res.logl_true = score_network(dataset.true_network_path, dataset.msa_path)
    res.n_reticulations_inferred, res.bic_inferred, res.logl_inferred = score_network(dataset.inferred_network_path, dataset.msa_path)
    _, res.bic_raxml, res.logl_raxml = score_network(dataset.raxml_tree_path, dataset.msa_path)
    network_1 = open(dataset.true_network_path).read()
    network_2 = open(dataset.inferred_network_path).read()
    res.topological_distances = get_dendro_scores(network_1, network_2)
    print(RESULT_CSV_HEADER+"\n" + res.get_csv_line() + "\n\n")
    return res
    
    
def run_inference_and_evaluate(datasets):
    for ds in datasets:
        infer_raxml_tree(ds)
        infer_network(ds.msa_path, ds.inferred_network_path, ds.timeout, ds.n_start_networks)
    results = [evaluate_dataset(ds) for ds in datasets]
    return results
    

def write_results_to_csv(results, csv_path):
    csv_file = open(csv_path, "w")
    header = DATASET_CSV_HEADER + ";" + RESULT_CSV_HEADER
    csv_file.write(header + "\n")
    for res in results:
        line = str(res.dataset.get_csv_line() + "," + res.get_csv_line() + "\n")
        csv_file.write(line)
    csv_file.close()


def run_experiments():
    pass


if __name__ == "__init__":
    run_experiments()
