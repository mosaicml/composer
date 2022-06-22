# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Script to generate graphs for repo README.md.

After generating images, upload to public hosting and update the README URLs.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

models = [{
    'name': 'GPT-2 125M',
    'baseline': 255,
    'performance': 150
}, {
    'name': 'ResNet-50',
    'baseline': 116,
    'performance': 15
}, {
    'name': 'DeepLab-v3',
    'baseline': 110,
    'performance': 36
}]


def generate_graph(filename, light_mode=True):
    """Generate Graphs.

    Args:
        filename (_type_): Name of output image SVG file
        light_mode (bool, optional): Render in light mode. Defaults to True.
    """
    font_color = 'black' if light_mode else 'white'
    mpl.rcParams['text.color'] = font_color
    mpl.rcParams['axes.labelcolor'] = font_color
    mpl.rcParams['xtick.color'] = font_color
    mpl.rcParams['ytick.color'] = font_color

    labels = [model['name'] for model in models]
    baselines = [model['baseline'] for model in models]
    performances = [model['performance'] for model in models]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.grid(which='major', axis='y')
    ax.set_axisbelow(True)

    rects1 = ax.bar(x - width / 2, baselines, width, label='Vanilla PyTorch', color=['#CCCCCC'])
    rects2 = ax.bar(x + width / 2, performances, width, label='Composer', color=['#EA4335'])

    ax.set_title('Cost comparison: Vanilla PyTorch vs. Composer')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)

    ax.bar_label(rects1, padding=3, fmt='$%g')
    ax.bar_label(rects2, padding=3, fmt='$%g')

    def format_cost(x, pos=None):
        return f'${int(x)}'

    ax.get_yaxis().set_major_formatter(format_cost)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_color('black' if light_mode else 'white')

    fig.tight_layout()

    plt.savefig(filename, transparent=True)


generate_graph('lightmode.svg', light_mode=True)
generate_graph('darkmode.svg', light_mode=False)
