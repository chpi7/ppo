import argparse
from os import altsep
import pathlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

COLOR = 'teal'

def main(input_path: pathlib.Path):

    data = pd.read_csv(input_path.as_posix())
    data['ValueStd'] = data['Value'].ewm(alpha=1-0.85, min_periods=1).std()
    data['ValueEMA'] = data['Value'].ewm(alpha=1-0.85, min_periods=1).mean()

    print(data)

    sns.set_theme(style='whitegrid')

    p = sns.lineplot(
        x='Step',
        y='ValueEMA',
        data=data,
        ci="sd",
        color=COLOR
    )

    plt.plot(
        data['Step'], 
        data['Value'],
        alpha=0.3,
        color=COLOR
    )

    plt.ylabel('Reward')
    plt.xlim(left=data['Step'].min(), right=data['Step'].max())
    plt.title('LunarLanderContinuous-v2')
    plt.tight_layout()

    plt.show()

    pass

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i','--input-path', help='Reward csv file to read from', type=pathlib.Path, default='./test1/rewards/LunarLanderCont-v2.csv')
    main(**vars(p.parse_args()))