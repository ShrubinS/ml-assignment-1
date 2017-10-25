CHUNKS = [100, 500, 1000, 5000, 10000, 50000, 100000]
# , 500000, 1000000, 5000000, 10000000, 50000000, 100000000]

DATASET_ROOT = '/Users/apple/projects/CS7CS4-machine-learning/Machine Learning Datasets'

DATA_SETS = {
    'sum_without_noise': {
        'location': DATASET_ROOT + '/the SUM dataset/without noise/The SUM dataset, without noise.csv',
        'header_present': True,
        'sep': ';',
        'classification_labels_present': True
    },
    'sum_with_noise': {
        'location': DATASET_ROOT + '/the SUM dataset/with noise/The SUM dataset, with noise.csv',
        'header_present': True,
        'sep': ';',
        'classification_labels_present': True
    },
    'twitter_buzz': {
        'location': DATASET_ROOT + '/Buzz in social media Data Set/Buzz in social media Data Set/Twitter/Twitter.data',
        'header_present': False,
        'sep': ',',
        'classification_labels_present': False  # Use quantiles to get 33rd, 67th percent quartiles and make classes for high, medium, low
    },
    '3d_road_network': {
        'location': DATASET_ROOT + '/3D Road Network/3D_spatial_network.txt',
        'header_present': False,
        'sep': ',',
        'classification_labels_present': False  # Use quantiles to get 3 classes again
    }
}