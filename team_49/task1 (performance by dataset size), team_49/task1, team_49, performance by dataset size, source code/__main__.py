import sys
import pandas as pd
import numpy as np
import config
import regression_algorithms


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    for name, location in config.DATA_SETS.items():
        df = pd.read_csv(location, sep=';')

        for chunk_size in config.CHUNKS:
            if chunk_size < len(df.index):
                print "chunk size", chunk_size
                df_chunk = df.head(chunk_size)
                regression_algorithms.perform_regression(df_chunk, name)




            # Do argument parsing here (eg. with argparse) and anything else
            # you want your project to do.


if __name__ == "__main__":
    main()
