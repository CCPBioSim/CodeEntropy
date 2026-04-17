import json

from tests.regression.cases import discover_cases


def main():
    systems = sorted({case.values[0].system for case in discover_cases()})
    print(json.dumps(systems))


if __name__ == "__main__":
    main()
