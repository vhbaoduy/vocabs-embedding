import pandas as pd
import os
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm


class DataPreparing(object):
    def __init__(self,
                 dataset_path,
                 labels,
                 output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.train = None
        self.valid = None
        self.test = None
        self.classes = labels
        self.n_classes = 0
        self.speakers = []
        self.n_speakers = 0

    def create_dataframe(self):
        print("Prepare data ...")
        valid = {
            "file_name": [],
            "speaker": [],
            "vocab": []
        }
        valid_lines = utils.load_txt(os.path.join(self.dataset_path, 'validation_list.txt'))
        # print(valid_lines)
        for line in valid_lines:
            parsing = line.split('/')
            vocab = parsing[0]
            if vocab in self.classes:
                file_name = line
                speaker = parsing[1].split('_')[0]
                valid["speaker"].append(speaker)
                valid["vocab"].append(vocab)
                valid["file_name"].append(file_name)

        test = {
            "file_name": [],
            "speaker": [],
            "vocab": []
        }
        test_lines = utils.load_txt(os.path.join(self.dataset_path, 'testing_list.txt'))
        for line in test_lines:
            parsing = line.split('/')
            vocab = parsing[0]
            file_name = line
            if vocab in self.classes:
                speaker = parsing[1].split('_')[0]
                test["speaker"].append(speaker)
                test["vocab"].append(vocab)
                test["file_name"].append(file_name)

        train = {
            "file_name": [],
            "speaker": [],
            "vocab": []
        }
        for idx, name in enumerate (self.classes):
            data_path = os.path.join(self.dataset_path, name)
            files = os.listdir(data_path)
            for i, file in enumerate(files):
                path = name + '/' + file
                # print(path)
                if file.endswith(".wav") and path not in test_lines and path not in valid_lines:
                    parsing = file.split("_")
                    speaker = parsing[0]
                    file_name = name + "/" + file
                    train["speaker"].append(speaker)
                    train["vocab"].append(name)
                    train["file_name"].append(file_name)

                    if speaker not in self.speakers:
                        self.speakers.append(speaker)

        self.n_speakers = len(self.speakers)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        df_test = pd.DataFrame(test)
        df_test.to_csv(os.path.join(self.output_path, 'test.csv'), index=False)
        df_valid = pd.DataFrame(valid)
        df_valid.to_csv(os.path.join(self.output_path, 'valid.csv'), index=False)

        df_train = pd.DataFrame(train)
        df_train.to_csv(os.path.join(self.output_path, 'train.csv'), index=False)
        print("Train: %d, Valid: %d, Test: %d"%(len(df_train), len(df_valid), len(df_test)))

if __name__ == '__main__':
    config = utils.load_config_file('../configs/configs.yaml')
    data_cfg = config['dataset']
    data_pre = DataPreparing(data_cfg['root_dir'],data_cfg['output_path'])
    data_pre.create_dataframe()