from dendroscope_wrapper import *
from netrax_wrapper import *
from raxml_wrapper import infer_raxml_tree, compute_rf_dist

from experiment_model import *

def retrieve_topological_distances(network_1_path, network_2_path):
    network_1 = open(network_1_path).read()
    network_2 = open(network_2_path).read()
    return get_dendro_scores(network_1, network_2)


def evaluate_dataset(dataset):
    res = Result(dataset)
    _, res.bic_true, res.logl_true = score_network(dataset.true_network_path, dataset.msa_path, dataset.likelihood_type)
    res.n_reticulations_inferred, res.bic_inferred, res.logl_inferred = score_network(dataset.inferred_network_path, dataset.msa_path, dataset.likelihood_type)
    _, res.bic_raxml, res.logl_raxml = score_network(dataset.raxml_tree_path, dataset.msa_path, dataset.likelihood_type)
    
    if dataset.start_from_raxml:
        res.n_reticulations_inferred_with_raxml, res.bic_inferred_with_raxml, res.logl_inferred_with_raxml = score_network(dataset.inferred_network_with_raxml_path, dataset.msa_path, dataset.likelihood_type)
        res.topological_distances_with_raxml = retrieve_topological_distances(dataset.true_network_path, dataset.inferred_network_with_raxml_path)
    
    res.topological_distances = retrieve_topological_distances(dataset.true_network_path, dataset.inferred_network_path)
    if dataset.n_reticulations == 0:
        res.rf_absolute_raxml, res.rf_relative_raxml = compute_rf_dist(dataset.true_network_path, dataset.raxml_tree_path)
    
    print(RESULT_CSV_HEADER+"\n" + res.get_csv_line() + "\n\n")
    return res
    
    
def run_inference_and_evaluate(datasets):
    for ds in datasets:
        infer_raxml_tree(ds)
        infer_network(ds)
    results = [evaluate_dataset(ds) for ds in datasets]
    return results
    

def write_results_to_csv(results, csv_path):
    csv_file = open(csv_path, "w")
    header = DATASET_CSV_HEADER + ";" + RESULT_CSV_HEADER
    csv_file.write(header + "\n")
    for res in results:
        line = str(res.dataset.get_csv_line() + "," + res.get_csv_line() + "\n")
        csv_file.write(line)
        if res.dataset.start_from_raxml:
            line2 = str(res.dataset.get_csv_line_with_raxml() + "," + res.get_csv_line_with_raxml() + "\n")
            csv_file.write(line2)
    csv_file.close()


def run_experiments():
    pass


if __name__ == "__init__":
    run_experiments()
