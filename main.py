import argparse
import os
import sys
import numpy as np
import torch
from utils.samplefiles import SampleFile

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(os.name)
    parser = argparse.ArgumentParser(description='Simple inference')
    parser.add_argument('--hdf-file-path',
                        help='Path to the HDF sample file (generated with '
                             'generate_sample.py) to be used. '
                             'Default: ./output/default.hdf.',
                        default='./output/default.hdf')
    print('Parsing command line arguments...', end=' ')
    arguments = vars(parser.parse_args())
    print('Done!')

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')
    hdf_file_path = str(arguments['hdf_file_path'])
    print('Reading in HDF file...', end=' ')
    data = SampleFile()
    data.read_hdf(hdf_file_path)
    df = data.as_dataframe(injection_parameters=True,
                           static_arguments=True)
    print('Done!')
    labels = []
    X = []
    # create x,labels
    for i in range(0, len(df)):
        sample = df.loc[i]
        X.append(sample.h1_strain)
        # check if there is injection, if yes there is a signal
        if isinstance(sample['h1_signal'], np.ndarray):
            labels.append(1)
        else:
            labels.append(0)
    print(f'Loaded a dataset of {labels.count(1)} injected and {labels.count(0)} background noise labels')
    print(f'Loaded a total of {len(X)} samples')
