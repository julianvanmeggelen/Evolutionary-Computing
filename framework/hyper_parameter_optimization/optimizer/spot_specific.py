import json

class SpotHyperDict():
    """Spot hyperparameter dictionary for neat on pole balancing."""

    def __init__(self):
        self.filename = "hyper_spot_generated.json"

    def load(self):
        with open(self.filename, "r") as f:
            d = json.load(f)
        return d

class DummyModel():
    def __init__(self):
        pass

    def __call__(self, name):
        return name
