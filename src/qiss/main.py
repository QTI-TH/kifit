import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--datapath", 
    default="data/nu.dat", 
    help="Path to target data.", 
    type=str
)

def main(datapath):
    """To be implemented"""

    # TO DO

    return -1

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)