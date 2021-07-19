from dendroscope_wrapper import *
from netrax_wrapper import infer_networks, score_network
from raxml_wrapper import infer_raxml_tree, compute_rf_dist

from experiment_model import *


def evaluate_dataset(ds):
    rf_absolute_raxml = -1
    rf_relative_raxml = -1
    if ds.n_reticulations == 0:
        rf_absolute_raxml, rf_relative_raxml = compute_rf_dist(
            ds.true_network_path, ds.raxml_tree_path)

    for var in ds.inference_variants:
        res = Result()
        _, res.bic_true, res.logl_true, res.aic_true, res.aicc_true = score_network(
            ds.true_network_path, ds.msa_path, ds.partitions_path, var.likelihood_type, var.brlen_linkage_type)
        _, res.bic_raxml, res.logl_raxml, res.aic_raxml, res.aicc_raxml = score_network(
            ds.raxml_tree_path, ds.msa_path, ds.partitions_path, var.likelihood_type, var.brlen_linkage_type)
        res.n_reticulations_inferred, res.bic_inferred, res.logl_inferred, res.aic_inferred, res.aicc_inferred = score_network(
            var.inferred_network_path, ds.msa_path, ds.partitions_path, var.likelihood_type, var.brlen_linkage_type)
        #res.topological_distances = retrieve_topological_distances(ds.true_network_path, var.inferred_network_path)

        if ds.n_reticulations == 0:
            res.rf_absolute_raxml = rf_absolute_raxml
            res.rf_relative_raxml = rf_relative_raxml
            if res.n_reticulations_inferred == 0:
                res.rf_absolute_inferred, res.rf_relative_inferred = compute_rf_dist(
                    ds.true_network_path, var.inferred_network_path)

        var.res = res


def run_inference_and_evaluate(datasets):
    for ds in datasets:
        ds.near_zero_branches_raxml = infer_raxml_tree(ds)
        infer_networks(ds)
        evaluate_dataset(ds)


def write_results_to_csv(datasets, csv_path):
    csv_file = open(csv_path, "w")
    header = DATASET_CSV_HEADER + "," + \
        INFERENCE_VARIANT_CSV_HEADER + "," + RESULT_CSV_HEADER
    csv_file.write(header + "\n")

    for ds in datasets:
        for var in ds.inference_variants:
            line = ds.get_csv_line() + "," + var.get_csv_line() + \
                "," + var.res.get_csv_line() + "\n"
            csv_file.write(line)

    csv_file.close()


if __name__ == "__init__":
    pass
