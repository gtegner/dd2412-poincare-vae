import sys
from argparse import Namespace
from main import main
import numpy as np
import pandas as pd
import os
from utils import ROOT_DIR, CSV_DIR
import configs.experiment_config as cfg
import argparse
import traceback


def run_experiment(experiment_name):

    failed = False
    save_dir = os.path.join(
        ROOT_DIR, CSV_DIR, experiment_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if experiment_name in ['toy-full']:
        df, experiments = cfg.toy_full(experiment_name)

    elif experiment_name in ['mnist']:
        df, experiments = cfg.mnist_experiment(experiment_name)

    num_runs = 5
    num_experiments = len(experiments)
    for ix, experiment in enumerate(experiments):
        print(f"Running Experiment {ix}...")
        print("\n")
        exp = Namespace(**experiment)
        print(exp)
        for seed in np.arange(num_runs):
            exp.seed = seed
            try:
                df = main(exp, df)
                print(
                    "Experiment [{}] {}/{} Done!".format(seed, ix, num_experiments))
                print(exp)
            except Exception as e:
                failed = True
                print("Experiment {} failed!".format(exp.experiment_name))
                print(e)
                print(traceback.format_exc())

    print("Experiment finished! Saving to csv")
    df.to_csv(os.path.join(
        save_dir, '{}-{}.csv'.format(experiment_name, str(failed))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    args = parser.parse_args()
    print(args.experiment)
    run_experiment(args.experiment)
