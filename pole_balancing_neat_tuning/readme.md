## Tuning with spot/optuna

To run this example using the spot tuner, with a timeout of 20 minutes:

```
python PYTHONPATH=. pole_balancing_neat_tuning/hyperopt.py --name tune_results --spot 1 --timeout 1200
```

To run with the Optuna tuner

```
python PYTHONPATH=. pole_balancing_neat_tuning/hyperopt.py --name tune_results--spot 0 --timeout 1200
```

To visualize the tuning results:

```
PYTHONPATH=. python pole_balancing_neat_tuning/plot_hyperopt.py --name tuner_results
```
