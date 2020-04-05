import argparse
import DEwNF
import pickle


def main(test, name):
    print(test)
    file_name = "name.pickle"
    with open(file_name, "wb") as f:
        pickle.dump(test, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="testing stuff")
    parser.add_argument("--name", help="Name of the experiment, used in naming outputs")
    args = parser.parse_args()
    main(args.test, name=args.name)
