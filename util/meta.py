import os
import csv

def create_custom_dataset(
    datapath,
    savepath,
    dataset_name="v0.1"
):
    """
    This function creates the csv file for a custom source separation dataset
    """

    mix_path = os.path.join(datapath, "mixture")
    s1_path = os.path.join(datapath, "source1")
    s2_path = os.path.join(datapath, "source2")
    files = os.listdir(mix_path)
    # files = sorted(files)

    mix_fl_paths = list()
    s1_fl_paths = list()
    s2_fl_paths = list()

    for fl in files:
        mix_fl_paths.append(os.path.join(mix_path, fl))
        s1 = fl.split('_')[1] + '_' + fl.split('_')[3]
        s2 = fl.split('_')[2] + '_' + fl.split('_')[4].split('.')[0]
        s1_fl_paths.append(os.path.join(s1_path, 'source_' + s1 + '.wav'))
        s2_fl_paths.append(os.path.join(s2_path, 'source_' + s2 + '.wav'))

    csv_columns = [
        "ID",
        "duration",
        "mix_wav",
        "mix_wav_format",
        "mix_wav_opts",
        "s1_wav",
        "s1_wav_format",
        "s1_wav_opts",
        "s2_wav",
        "s2_wav_format",
        "s2_wav_opts",
        "noise_wav",
        "noise_wav_format",
        "noise_wav_opts",
    ]

    with open(
        os.path.join(savepath, dataset_name + "_train.csv"), "w"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for i, (mix_path, s1_path, s2_path) in enumerate(
            zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
        ):

            row = {
                "ID": i,
                "duration": 1.0,
                "mix_wav": mix_path,
                "mix_wav_format": "wav",
                "mix_wav_opts": None,
                "s1_wav": s1_path,
                "s1_wav_format": "wav",
                "s1_wav_opts": None,
                "s2_wav": s2_path,
                "s2_wav_format": "wav",
                "s2_wav_opts": None,
            }
            writer.writerow(row)
