import pandas as pd
import os
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm


class DataPreparing(object):
    def __init__(self, dataset_path,labels, output_path):
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
        # audio_path = os.path.join(self.dataset_path, 'audio')
        # Insert the name of class
        # for folder in os.listdir(self.dataset_path):
        #     if os.path.isdir(os.path.join(self.dataset_path, folder)) and folder != '_background_noise_':
        #         self.classes.append(folder)
        # self.n_classes = len(self.classes)

        valid = {
            "file_name": [],
            "speaker": [],
            "vocab": []
        }
        valid_lines = utils.load_txt(os.path.join(self.dataset_path, 'validation_list.txt'))
        # print(valid_lines)
        valid_process = tqdm(valid_lines)
        print("Prepare validation file...")
        for line in valid_process:
            parsing = line.split('/')
            file_name = line
            vocab = parsing[0]
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
        test_process = tqdm(test_lines)
        print("Prepare test file...")
        for line in test_process:
            parsing = line.split('/')
            file_name = line
            vocab = parsing[0]
            speaker = parsing[1].split('_')[0]
            test["speaker"].append(speaker)
            test["vocab"].append(vocab)
            test["file_name"].append(file_name)

        train = {
            "file_name": [],
            "speaker": [],
            "vocab": []
        }
        print("Prepare training file...")
        for idx, name in enumerate (self.classes):
            print("{}. Label {}".format(idx, name))
            data_path = os.path.join(self.dataset_path, name)
            files = os.listdir(data_path)
            file_process = tqdm(files)
            for i, file in enumerate(file_process):
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
        # classes = sorted(self.classes)
        # s = ', '.join([str(x) for x in classes])
        # print(s)
        # Create dataframe
        # Store dataframe
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        df_test = pd.DataFrame(test)
        df_test.to_csv(os.path.join(self.output_path, 'test.csv'), index=False)
        self.test = df_test
        df_valid = pd.DataFrame(valid)
        df_valid.to_csv(os.path.join(self.output_path, 'valid.csv'), index=False)
        self.valid = df_valid

        df_train = pd.DataFrame(train)
        df_train.to_csv(os.path.join(self.output_path, 'train.csv'), index=False)

        print(len(df_train), len(df_test), len(df_valid))

    def print_dataset_statistics(self):
        print("#" * 5, "DATASET STATISTICS", "#" * 5)
        print('The number of classes: {}'.format(self.n_classes))
        print('The number of speakers: {}'.format(self.n_speakers))
        print('The number of rows in dataframe: {} (Train {} Valid {})'.format(len(self.train) + len(self.valid),
                                                                               len(self.train),
                                                                               len(self.valid)))
        print("#" * 50)

    def visualize_dataset_statistics(self, df, path):
        # df = pd.concat([self.train, self.valid], ignore_index=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))
        class_counter = df.groupby(['vocab'])['vocab'].count()
        ax1.bar(class_counter.index.values.tolist(), class_counter, width=0.3)
        ax1.set_title('Vocab bar chart')
        ax1.set_ylabel('Count')

        speaker_counter = df.groupby(['speaker'])['speaker'].count().reset_index(name='counts')
        ax2.boxplot(speaker_counter['counts'])
        ax2.set_title('Speaker Boxplot')
        ax2.set_xlabel('Speaker')
        ax2.set_ylabel('Count')

        fig.savefig(os.path.join(self.output_path, path))
        # plt.show()


if __name__ == '__main__':
    config = utils.load_config_file('../configs/configs.yaml')
    data_cfg = config['dataset']
    data_pre = DataPreparing(data_cfg['root_dir'],data_cfg['output_path'])
    data_pre.create_dataframe()