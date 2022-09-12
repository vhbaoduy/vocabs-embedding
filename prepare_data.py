import argparse
from datasets import DataPreparing
from utils import load_config_file
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare dataframe for training, validation and test.")
    parser.add_argument(
        "--config_file", default="configs.yaml", help="path to config file", type=str, required=True)

    args = parser.parse_args()
    config_path = args.config_file
    path = os.path.join("./configs", config_path)
    dataset_cfgs = load_config_file(path)['Dataset']
    data_preparing = DataPreparing(dataset_cfgs['root_dir'],
                                   dataset_cfgs['labels'],
                                   dataset_cfgs['output_path'])
    data_preparing.create_dataframe()