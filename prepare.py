import argparse
from datasets import *

def main():
    parser = argparse.ArgumentParser(description="Prepare and visualize dataset statistics")
    parser.add_argument(
        "-config_file", default="configs.yaml", help="path to config file", type=str
    )
    parser.add_argument(
        "-create_all", help="optional for create dataframe", type=bool,default=True
    )
    args = parser.parse_args()
    config_file = args.config_file
    create_all = args.create_all

    configs = utils.load_config_file(os.path.join('./configs', config_file))
    dataset_cfgs = configs['Dataset']

    # Prepare data
    data_preparing = DataPreparing(dataset_cfgs['root_dir'],
                                   dataset_cfgs['labels'],
                                   dataset_cfgs['output_path'],
                                   create_all=create_all)
    data_preparing.create_dataframe()


if __name__ == '__main__':
    main()