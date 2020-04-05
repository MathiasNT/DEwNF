import argparse
import DEwNF


def main(test):
    print(test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="testing stuff")
    args = parser.parse_args()
    main(args.test)
