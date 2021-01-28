from create_plots import create_plots
import argparse


def parse_command_line_arguments_postprocess():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="small_network")
    args = CLI.parse_args()
    return args.prefix


if __name__ == "__main__":
    prefix = parse_command_line_arguments_postprocess()
    create_plots(prefix)
