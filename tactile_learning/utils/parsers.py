import argparse

def choose_preprocess():
    '''
    Choose the detector that needs to be used
    '''
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-m',
    #     '--method',
    #     default='VR',
    #     nargs='?',
    #     choices=['VR', 'MP'],
    #     help='choose between the detection methods to use (default: %(default)s).'
    # )

    parser.add_argument(
        '-p',
        '--preprocess',
        action='store_true'
    )  

    return parser.parse_args()