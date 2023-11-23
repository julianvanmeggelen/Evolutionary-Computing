import json

class SpotHyperDict():
    """Spot hyperparameter dictionary for neat on pole balancing."""

    def __init__(self):
        self.filename = "hyper_spot.json"

    def load(self):
        with open(self.filename, "r") as f:
            d = json.load(f)
        return d