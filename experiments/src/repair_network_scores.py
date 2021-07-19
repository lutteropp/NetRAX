from netrax_wrapper import score_network
from experiment_model import LikelihoodType, BrlenLinkageType
import pandas as pd
import argparse


def recompute_network_scores(prefix):
    input_csv_path = prefix + "_results.csv"
    output_csv_path = prefix + "_results.csv"
    df = pd.read_csv(input_csv_path)
    
    bic_true = []
    logl_true = []
    aic_true = []
    aicc_true = []

    bic_inferred = []
    logl_inferred = []
    aic_inferred = []
    aicc_inferred = []
        
    for _, row in df.iterrows():
        true_network_path = prefix + "/" + row["true_network_path"]
        msa_path = true_network_path.split("_msa.txt")[0] + "_msa.txt"
        raxml_tree_path = true_network_path.split("_msa.txt")[0] + ".raxml.bestTree"
        partitions_path = true_network_path.split("_msa.txt")[0] + "_partitions.txt"

        if row["use_partitioned_msa"].contains("False") or row["use_partitioned_msa"].contains("false") or row["use_partitioned_msa"].contains("FALSE"):
            partitions_path = "DNA"

        inferred_network_path = prefix + "/" + row["inferred_network_path"]

        likelihood_type_str = row["likelihood_type"]
        likelihood_type = LikelihoodType.AVERAGE
        if likelihood_type_str.contains("BEST") or likelihood_type_str.contains("Best") or likelihood_type_str.contains("best"):
            likelihood_type = LikelihoodType.BEST

        brlen_linkage_type_str = row["brlen_linkage_type"]
        brlen_linkage_type = BrlenLinkageType.LINKED
        if brlen_linkage_type_str.contains("UNLINKED") or brlen_linkage_type_str.contains("Unlinked") or brlen_linkage_type_str.contains("unlinked"):
            brlen_linkage_type = BrlenLinkageType.UNLINKED
        elif brlen_linkage_type_str.contains("SCALED") or brlen_linkage_type_str.contains("Scaled") or brlen_linkage_type_str.contains("scaled"):
            brlen_linkage_type = BrlenLinkageType.SCALED

        _, act_bic_true, act_logl_true, act_aic_true, act_aicc_true = score_network(
            true_network_path, msa_path, partitions_path, likelihood_type, brlen_linkage_type)
        bic_true.append(act_bic_true)
        logl_true.append(act_logl_true)
        aic_true.append(act_aic_true)
        aicc_true.append(act_aicc_true)

        _, act_bic_raxml, act_logl_raxml, act_aic_raxml, act_aicc_raxml = score_network(
            raxml_tree_path, msa_path, partitions_path, likelihood_type, brlen_linkage_type)
        bic_raxml.append(act_bic_raxml)
        logl_raxml.append(act_logl_raxml)
        aic_raxml.append(act_aic_raxml)
        aicc_raxml.append(act_aicc_raxml)

        _, act_bic_inferred, act_logl_inferred, act_aic_inferred, act_aicc_inferred = score_network(
            inferred_network_path, msa_path, partitions_path, likelihood_type, brlen_linkage_type)
        bic_inferred.append(act_bic_inferred)
        logl_inferred.append(act_logl_inferred)
        aic_inferred.append(act_aic_inferred)
        aicc_inferred.append(act_aicc_inferred)
        
        n_equal_tree_pairs.append(n_equal)
        if n_pairs > 0:
            true_network_weirdness.append(float(n_equal)/n_pairs)
        else:
            true_network_weirdness.append(0.0)
            
        print(row["name"])
        
    df['bic_true'] = bic_true
    df['logl_true'] = logl_true
    df['aic_true'] = aic_true
    df['aicc_true'] = aicc_true

    df['bic_raxml'] = bic_raxml
    df['logl_raxml'] = logl_raxml
    df['aic_raxml'] = aic_raxml
    df['aicc_raxml'] = aicc_raxml

    df['bic_inferred'] = bic_inferred
    df['logl_inferred'] = logl_inferred
    df['aic_inferred'] = aic_inferred
    df['aicc_inferred'] = aicc_inferred

    df.to_csv(output_csv_path, index=False)


def parse_command_line_arguments_scores():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="")
    args = CLI.parse_args()
    return args.prefix


if __name__ == "__main__":
    prefix = parse_command_line_arguments_scores()
    recompute_network_scores(prefix)
    
