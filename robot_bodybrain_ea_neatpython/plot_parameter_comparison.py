import argparse
from hyper_parameter_optimization.result.optimization_result import OptimizationResult
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

def get_params(trials, result, label):
    res = []
    for trial in trials:
        for param in result.tune_params:
            res.append(
                {'Parameter': param, 'Value': getattr(trial.config,param), 'Label': label}
            )
    return res


def get_df(result):
    best_10 = sorted(result.runs, key = lambda run: run.utility)[-10:]
    worst_10 = sorted(result.runs, key = lambda run: run.utility)[:10]

    best_10_params = get_params(best_10, result, label='Top 10')
    worst_10_params = get_params(worst_10, result, label='Bottom 10')

    df = pd.DataFrame(best_10_params + worst_10_params)
    return df


def plot_params(result: OptimizationResult):
    df = get_df(result)
    sns.catplot(
        df[df['Parameter'] != 'activation_default'],
        x='Parameter',
        y='Value',
        hue = 'Label'
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)  #path to save result
    parser.add_argument('--baseline', type=str)  #path to save result

    args = parser.parse_args()
    plot_save_path = '/'.join(args.name.split('/')[:-1])

    #Load result
    optimization_result = OptimizationResult.load(args.name)
    plot_params(optimization_result)
    plt.show()

