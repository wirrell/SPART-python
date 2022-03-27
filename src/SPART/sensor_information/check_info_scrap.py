import pickle
import pprint
from pathlib import Path

pp = pprint.PrettyPrinter()


for sensor in Path(".").glob("*.pkl"):
    with open(sensor, "rb") as f:
        info = pickle.load(f)
        print(sensor)
        pp.pprint(info.keys())
