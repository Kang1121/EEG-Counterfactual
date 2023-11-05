# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import yaml
import numpy as np

from utils.common_config import get_test_dataset, get_vis_transform, get_customized_dataset
from utils.path import Path
from utils.visualize import visualize_counterfactuals
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from explainer.eval import compute_eval_metrics
parser = argparse.ArgumentParser(description="Visualize counterfactual explanations")
parser.add_argument("--config_path", type=str, required=True)


def main(iii=None):
    args = parser.parse_args()
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)

    experiment_name = os.path.basename(args.config_path).split(".")[0]
    dirpath = os.path.join(Path.output_root_dir(), experiment_name)
    save_dir = experiment_name + '/'

    # dataset = get_test_dataset(get_vis_transform(), return_image_only=True)

    # dataset = get_customized_dataset(config['data_path'], return_image_only=True)  # plot
    # dataset = get_customized_dataset(config['data_path'])  # Compute eval metrics
    if iii is not None:
        counterfactuals = np.load(
            os.path.join(dirpath, "counterfactuals_with_pretrained_{}.npy".format(iii)), allow_pickle=True
        ).item()
    else:
        counterfactuals = np.load(
            os.path.join(dirpath, "counterfactuals.npy"), allow_pickle=True
        ).item()

    # Compute eval metrics
    # result = compute_eval_metrics(
    #     counterfactuals,
    #     dataset=dataset,
    # )
    #
    # print("Eval results single edit: {}".format(result["single_edit"]))
    # print("Eval results all edits: {}".format(result["all_edit"]))
    # average_num_edits = np.mean([len(res["edits"]) for res in counterfactuals.values()])
    # print("Average number of edits is {:.2f}".format(average_num_edits))

    marginal_histogram(counterfactuals, iii, save_dir)


    for idx in np.random.choice(list(counterfactuals.keys()), 5):
        cf = counterfactuals[idx]
        swap_path = save_dir + 'swap_plot/pretrained_{}/'.format(iii) if iii is not None else save_dir + 'swap_plot'
        if not os.path.exists(swap_path):
            os.makedirs(swap_path)
        visualize_counterfactuals(
            edits=cf["edits"],
            query_index=cf["query_index"],
            distractor_index=cf["distractor_index"],
            dataset=dataset,
            n_pix=7,
            fname= swap_path + 'example_{}.png'.format(idx)
        )


def marginal_histogram(data, fname=None, save_dir=None):

    sns.set_theme(style="white")
    plt.rcParams['figure.dpi'] = 300
    distractor_list, query_list = [], []
    for key in data.keys():
        edits = data[key]['edits']
        for edit in edits:
            distractor_list.append(edit[1] % 49)
            query_list.append(edit[0])
    df = pd.DataFrame({'Distractor': distractor_list, 'Query': query_list})

    x_range, y_range = np.arange(1, 50), np.arange(1, 50)
    xx, yy = zip(*product(x_range, y_range))

    artificial_df = pd.DataFrame({'Query': xx, 'Distractor': yy})
    combined_df = pd.concat([df, artificial_df], ignore_index=True)
    weights = np.concatenate([np.ones(len(distractor_list)), np.full_like(xx, 1e-10)])

    g = sns.JointGrid(data=combined_df, x="Query", y="Distractor", space=0)
    g.plot_joint(sns.kdeplot, fill=True, weights=weights, thresh=0, cmap="rocket")
    g.ax_joint.set_xlim(0, 49), g.ax_joint.set_ylim(0, 49)
    g.ax_joint.set_xticks([]), g.ax_joint.set_yticks([])
    g.x, g.y = g.x[:query_list.__len__()], g.y[:query_list.__len__()]
    g.plot_marginals(sns.histplot, data=df, color="#03051A", alpha=1, bins=49)

    regions = {
        'c': [(0, 10, 5.5, r'$\gamma$')],
        'b': [(10, 39, 25, r'$\beta$')],
        'r': [(39, 44, 41.5, r'$\alpha$')],
        'g': [(44, 49, 46.5, r'$\theta$ & $\delta$')]
    }

    for color, areas in regions.items():
        for start, end, label_pos, label in areas:
            add_span_and_label(g.ax_joint, 'x', start, end, label_pos, label, color)
            add_span_and_label(g.ax_joint, 'y', start, end, label_pos, label, color, text_rotation=90)
    # g.ax_joint.axvspan(0, 10, ymin=-0.05, ymax=0, facecolor='c', alpha=0.3, clip_on=False)
    # g.ax_joint.text(5.5, -1.2, r'$\gamma$', ha='center', va='center', color='k', transform=g.ax_joint.transData, fontsize=9)
    # g.ax_joint.axvspan(10, 39, ymin=-0.05, ymax=0, facecolor='b', alpha=0.3, clip_on=False)
    # g.ax_joint.text(25, -1.2, r'$\beta$', ha='center', va='center', color='k', transform=g.ax_joint.transData, fontsize=9)
    # g.ax_joint.axvspan(39, 44, ymin=-0.05, ymax=0, facecolor='r', alpha=0.3, clip_on=False)
    # g.ax_joint.text(41.5, -1.2, r'$\alpha$', ha='center', va='center', color='k', transform=g.ax_joint.transData, fontsize=9)
    # g.ax_joint.axvspan(44, 49, ymin=-0.05, ymax=0, facecolor='g', alpha=0.3, clip_on=False)
    # g.ax_joint.text(46.5, -1.2, r'$\theta$ & $\delta$', ha='center', va='center', color='k', transform=g.ax_joint.transData, fontsize=9)
    #
    # # Create a span on the left y-axis
    # g.ax_joint.axhspan(0, 10, xmin=-0.05, xmax=0, facecolor='c', alpha=0.3, clip_on=False)
    # g.ax_joint.text(-1.2, 5.5, r'$\gamma$', ha='center', va='center', color='k', rotation=90,
    #                 transform=g.ax_joint.transData, fontsize=9)
    # g.ax_joint.axhspan(10, 39, xmin=-0.05, xmax=0, facecolor='b', alpha=0.3, clip_on=False)
    # g.ax_joint.text(-1.2, 25, r'$\beta$', ha='center', va='center', color='k', rotation=90,
    #                 transform=g.ax_joint.transData, fontsize=9)
    # g.ax_joint.axhspan(39, 44, xmin=-0.05, xmax=0, facecolor='r', alpha=0.3, clip_on=False)
    # g.ax_joint.text(-1.2, 41.5, r'$\alpha$', ha='center', va='center', color='k', rotation=90,
    #                 transform=g.ax_joint.transData, fontsize=9)
    # g.ax_joint.axhspan(44, 49, xmin=-0.05, xmax=0, facecolor='g', alpha=0.3, clip_on=False)
    # g.ax_joint.text(-1.2, 46.5, r'$\theta$ & $\delta$', ha='center', va='center', color='k', rotation=90,
    #                 transform=g.ax_joint.transData, fontsize=9)

    g.ax_joint.xaxis.set_label_coords(0.5, -0.07)
    g.ax_joint.yaxis.set_label_coords(-0.07, 0.5)
    g.ax_joint.set_xlabel('Span of patches in query set', fontsize=11)
    g.ax_joint.set_ylabel('Span of patches in distractor set', fontsize=11)
    # g.fig.suptitle('Marginal histogram of patches used in the edits', y=1, fontsize=15)

    if fname is None:
        plt.show()
    else:
        save_path = save_dir + 'JointGrid/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + 'pretrained_{}.png'.format(fname), dpi=300)
    plt.close()


def add_span_and_label(ax, orientation, start, end, label_pos, label, color, alpha=0.3, text_rotation=0):
    # Determine if we're working on x or y axis
    if orientation == 'x':
        ax.axvspan(start, end, ymin=-0.05, ymax=0, facecolor=color, alpha=alpha, clip_on=False)
        ax.text(label_pos, -1.2, label, ha='center', va='center', color='k', transform=ax.transData,
                fontsize=9, rotation=text_rotation)
    elif orientation == 'y':
        ax.axhspan(start, end, xmin=-0.05, xmax=0, facecolor=color, alpha=alpha, clip_on=False)
        ax.text(-1.2, label_pos, label, ha='center', va='center', color='k', transform=ax.transData,
                fontsize=9, rotation=text_rotation)


if __name__ == "__main__":
    for i in range(12, 14):
        main(i)
