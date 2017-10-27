import sys
import pandas as pd
import numpy as np
import config
import regression_algorithms
import classification_algorithms
import shutil
import csv
import tempfile
import zipfile


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    out_dict = {}

    for name, data_set in config.DATA_SETS.items():
        if data_set['zipped']:
            zip_ref = zipfile.ZipFile(data_set['location'], 'r')
            zip_ref.extractall(config.TEMP_DIR)
            zip_ref.close()
            data_set['location'] = config.TEMP_DIR + 'Buzz in social media Data Set/Twitter/Twitter.data'

        if data_set['header_present']:
            df = pd.read_csv(data_set['location'], sep=data_set['sep'])
        else:
            df = pd.read_csv(data_set['location'], sep=data_set['sep'], header=None)

        for chunk_size in config.CHUNKS:
            df_size = len(df.index)
            if (chunk_size <= df_size) or ((chunk_size > df_size) and (df_size/chunk_size >= 0.9)):
                print "chunk size", chunk_size
                df_chunk = df.head(chunk_size)
                print "dataset", name
                regression_algorithms.perform_regression(df_chunk, name, chunk_size, out_dict)

        for chunk_size in config.CHUNKS:
            df_size = len(df.index)
            if (chunk_size <= df_size) or ((chunk_size > df_size) and (df_size / chunk_size >= 0.9)):
                print "chunk size", chunk_size
                df_chunk = df.head(chunk_size)
                print "dataset", name
                classification_algorithms.perform_classification(df_chunk, name, chunk_size, out_dict)

        with open('output/output.csv', 'wb') as my_file:
            for algo, value_map in out_dict.iteritems():
                row1 = list()
                row2 = list()
                row1.append(algo)
                row2.append(algo)
                for size, value in value_map.iteritems():
                    row1.append(str(value))
                    row2.append(str(size))
                my_file.write(','.join(row1) + '\n')
                my_file.write(','.join(row2) + '\n')


if __name__ == "__main__":
    main()
