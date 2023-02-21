import sys
import os
import argparse
import time
import numpy as np
if __name__ == "__main__":
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')
    print('')
    print('GENERATING IMAGE REPRESENTATION OF THE DATASET WITH LABELS')
    print('')
    #add parser
    parser = argparse.ArgumentParser(description='Generate image data from hdf file')