# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def visualize_counterfactuals(
    edits,
    query_index,
    distractor_index,
    dataset,
    n_pix,
    fname=None,
):
    # load image
    query_img = dataset.__getitem__(query_index).mean(axis=0)
    height, width = query_img.shape[0], query_img.shape[1]

    # geometric properties of cells
    width_cell = width // n_pix
    height_cell = height // n_pix

    # create plot
    n_edits = len(edits)
    fig, axes = plt.subplots(n_edits, 2, figsize=(10, 4 * n_edits), dpi=300)

    if n_edits == 1:
        axes = axes[np.newaxis ,:]
    # fig.text(0.05, 0.75, "Query", rotation=90, verticalalignment='center', fontsize=14)
    # fig.text(0.05, 0.25, "Distractor", rotation=90, verticalalignment='center', fontsize=14)

    x_ticks_pixel = np.linspace(0, 224, 7)  # Assuming 6 ticks for [0, 1, 2, 3]
    x_ticks_label = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    y_ticks_pixel = np.linspace(0, 224, 9)  # Assuming 9 ticks for [0, 5, 10, ..., 40]
    y_ticks_label = np.arange(40, -1, -5)

    # loop over edits
    for ii, edit in enumerate(edits):
        # show query
        cell_index_query = edit[0]
        row_index_query = cell_index_query // n_pix
        col_index_query = cell_index_query % n_pix

        query_left_box = int(col_index_query * width_cell)
        query_top_box = int(row_index_query * height_cell)

        rect = patches.Rectangle(
            (query_left_box, query_top_box),
            width_cell,
            height_cell,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            zorder=2,
        )

        # axes[ii][0].imshow(query_img)
        img_display = axes[ii][0].imshow(np.abs(query_img), aspect='auto', cmap='jet', interpolation='bilinear')
        fig.colorbar(img_display, ax=axes[ii][0], label='Magnitude')

        axes[ii][0].add_patch(rect)
        axes[ii][0].set_xticks(x_ticks_pixel)
        axes[ii][0].set_xticklabels(x_ticks_label)
        axes[ii][0].set_yticks(y_ticks_pixel)
        axes[ii][0].set_yticklabels(y_ticks_label)
        axes[ii][0].set_ylabel("Frequency (Hz)")
        axes[ii][0].set_xlabel("Time (s)")
        if ii == 0:
            axes[ii][0].set_title("Query", fontsize=14)

        # show distractor
        cell_index_distractor = edit[1]

        index_distractor = distractor_index[cell_index_distractor // (n_pix**2)]
        img_distractor = dataset.__getitem__(index_distractor).mean(axis=0)

        cell_index_distractor = cell_index_distractor % (n_pix**2)
        row_index_distractor = cell_index_distractor // n_pix
        col_index_distractor = cell_index_distractor % n_pix

        distractor_left_box = int(col_index_distractor * width_cell)
        distractor_top_box = int(row_index_distractor * height_cell)

        rect = patches.Rectangle(
            (distractor_left_box, distractor_top_box),
            width_cell,
            height_cell,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            zorder=2,
        )

        img_display = axes[ii][1].imshow(np.abs(img_distractor), aspect='auto', cmap='jet', interpolation='bilinear')
        fig.colorbar(img_display, ax=axes[ii][1], label='Magnitude')

        axes[ii][1].add_patch(rect)
        axes[ii][1].set_xticks(x_ticks_pixel)
        axes[ii][1].set_xticklabels(x_ticks_label)
        axes[ii][1].set_yticks(y_ticks_pixel)
        axes[ii][1].set_yticklabels(y_ticks_label)
        axes[ii][1].set_ylabel("Frequency (Hz)")
        axes[ii][1].set_xlabel("Time (s)")
        if ii == 0:
            axes[ii][1].set_title("Distractor", fontsize=14)

    # save or view
    plt.tight_layout()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
