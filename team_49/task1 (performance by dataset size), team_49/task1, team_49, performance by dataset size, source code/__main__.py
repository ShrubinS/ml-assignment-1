import sys
import re
import pandas as pd
import numpy as np
import config
import regression_algorithms
import classification_algorithms


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    for name, data_set in config.DATA_SETS.items():
        if data_set['header_present']:
            df = pd.read_csv(data_set['location'], sep=data_set['sep'])
        else:
            df = pd.read_csv(data_set['location'], sep=data_set['sep'], header=None)

        for chunk_size in config.CHUNKS:
            if chunk_size < len(df.index):
                print "chunk size", chunk_size
                df_chunk = df.head(chunk_size)
                print "dataset", name
                regression_algorithms.perform_regression(df_chunk, name, chunk_size)
                classification_algorithms.perform_classification(df_chunk, name, chunk_size)


if __name__ == "__main__":
    main()
