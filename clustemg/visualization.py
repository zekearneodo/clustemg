import seaborn as sns
import numpy as np
import umap


from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import cKDTree



from matplotlib import lines
from matplotlib import gridspec

from matplotlib import pyplot as plt
from matplotlib import path as mpath
from matplotlib import collections as mc
from matplotlib import collections as mcoll


from matplotlib.lines import Line2D
from tqdm.autonotebook import tqdm

from PIL import Image
import io

def scatter_projections(
    syllables=None,
    projection=None,
    labels=None,
    ax=None,
    figsize=(10, 10),
    alpha=0.1,
    s=1,
    color="k",
    color_palette="tab20",
    show_legend=True,
):
    """ creates a scatterplot of syllables using some projection
    """
    if projection is None:
        if syllables is None:
            raise ValueError("Either syllables or projections must by passed")

        syllables_flattened = np.reshape(
            syllables, (np.shape(syllables)[0], np.prod(np.shape(syllables)[1:]))
        )

        # if no projection is passed, assume umap
        fit = umap.UMAP(min_dist=0.25, verbose=True)
        u_all = fit.fit_transform(syllables_flattened)

    # color labels
    if labels is not None:
        pal = sns.color_palette(color_palette, n_colors=len(np.unique(labels)))
        lab_dict = {lab: pal[i] for i, lab in enumerate(np.unique(labels))}
        colors = np.array([lab_dict[i] for i in labels])
    else:
        colors = color

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

        # plot
    ax.scatter(projection[:, 0], projection[:, 1], alpha=alpha, s=s, color=colors)
    if labels is not None:
        legend_elements = [
            Line2D([0], [0], marker="o", color=value, label=key)
            for key, value in lab_dict.items()
        ]
        if show_legend:
            ax.legend(handles=legend_elements)
    return ax


def scatter_spec(
    z,
    specs,
    column_size=10,
    pal_color="hls",
    matshow_kwargs={"cmap": plt.cm.Greys},
    scatter_kwargs={"alpha": 0.5, "s": 1},
    line_kwargs={"lw": 1, "ls": "dashed", "alpha": 1},
    color_points=False,
    figsize=(10, 10),
    range_pad=0.1,
    x_range=None,
    y_range=None,
    enlarge_points=0,
    draw_lines=True,
    n_subset=-1,
    ax=None,
    show_scatter=True,
    border_line_width = 1,
):
    """
    """
    n_columns = column_size * 4 - 4
    pal = sns.color_palette(pal_color, n_colors=n_columns)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(column_size, column_size)

    if x_range is None and y_range is None:
        xmin, xmax = np.sort(np.vstack(z)[:, 0])[
            np.array([int(len(z) * 0.01), int(len(z) * 0.99)])
        ]
        ymin, ymax = np.sort(np.vstack(z)[:, 1])[
            np.array([int(len(z) * 0.01), int(len(z) * 0.99)])
        ]
        # xmin, ymin = np.min(z, axis=0)
        # xmax, ymax = np.max(z, axis=0)
        xmin -= (xmax - xmin) * range_pad
        xmax += (xmax - xmin) * range_pad
        ymin -= (ymax - ymin) * range_pad
        ymax += (ymax - ymin) * range_pad
    else:
        xmin, xmax = x_range
        ymin, ymax = y_range

    x_block = (xmax - xmin) / column_size
    y_block = (ymax - ymin) / column_size

    # ignore segments outside of range
    z = np.array(z)
    mask = np.array(
        [(z[:, 0] > xmin) & (z[:, 1] > ymin) & (z[:, 0] < xmax) & (z[:, 1] < ymax)]
    )[0]

    if "labels" in scatter_kwargs:
        scatter_kwargs["labels"] = np.array(scatter_kwargs["labels"])[mask]
    specs = np.array(specs)[mask]
    z = z[mask]

    # prepare the main axis
    main_ax = fig.add_subplot(gs[1 : column_size - 1, 1 : column_size - 1])
    # main_ax.scatter(z[:, 0], z[:, 1], **scatter_kwargs)
    if show_scatter:
        scatter_projections(projection=z, ax=main_ax, **scatter_kwargs)

    # loop through example columns
    axs = {}
    for column in range(n_columns):
        # get example column location
        if column < column_size:
            row = 0
            col = column

        elif (column >= column_size) & (column < (column_size * 2) - 1):
            row = column - column_size + 1
            col = column_size - 1

        elif (column >= ((column_size * 2) - 1)) & (column < (column_size * 3 - 2)):
            row = column_size - 1
            col = column_size - 3 - (column - column_size * 2)
        elif column >= column_size * 3 - 3:
            row = n_columns - column
            col = 0

        axs[column] = {"ax": fig.add_subplot(gs[row, col]), "col": col, "row": row}
        # label subplot
        """axs[column]["ax"].text(
            x=0.5,
            y=0.5,
            s=column,
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[column]["ax"].transAxes,
        )"""

        # sample a point in z based upon the row and column
        xpos = xmin + x_block * col + x_block / 2
        ypos = ymax - y_block * row - y_block / 2
        # main_ax.text(x=xpos, y=ypos, s=column, color=pal[column])

        axs[column]["xpos"] = xpos
        axs[column]["ypos"] = ypos

    main_ax.set_xlim([xmin, xmax])
    main_ax.set_ylim([ymin, ymax])

    # create a voronoi diagram over the x and y pos points
    points = [[axs[i]["xpos"], axs[i]["ypos"]] for i in axs.keys()]

    voronoi_kdtree = cKDTree(points)
    vor = Voronoi(points)

    # plot voronoi
    # voronoi_plot_2d(vor, ax = main_ax);

    # find where each point lies in the voronoi diagram
    z = z[:n_subset]
    point_dist, point_regions = voronoi_kdtree.query(list(z))

    lines_list = []
    # loop through regions and select a point
    for key in axs.keys():
        # sample a point in (or near) voronoi region
        nearest_points = np.argsort(np.abs(point_regions - key))
        possible_points = np.where(point_regions == point_regions[nearest_points][0])[0]
        chosen_point = np.random.choice(a=possible_points, size=1)[0]
        point_regions[chosen_point] = 1e4
        # plot point
        if enlarge_points > 0:
            if color_points:
                color = pal[key]
            else:
                color = "k"
            main_ax.scatter(
                [z[chosen_point, 0]],
                [z[chosen_point, 1]],
                color=color,
                s=enlarge_points,
            )
        # draw spec
        axs[key]["ax"].matshow(
            specs[chosen_point],
            origin="lower",
            interpolation="none",
            aspect="auto",
            **matshow_kwargs,
        )

        axs[key]["ax"].set_xticks([])
        axs[key]["ax"].set_yticks([])
        if color_points:
            plt.setp(axs[key]["ax"].spines.values(), color=pal[key])

        for i in axs[key]["ax"].spines.values():
            i.set_linewidth(border_line_width) 

        # draw a line between point and image
        if draw_lines:
            mytrans = (
                axs[key]["ax"].transAxes + axs[key]["ax"].figure.transFigure.inverted()
            )

            line_end_pos = [0.5, 0.5]

            if axs[key]["row"] == 0:
                line_end_pos[1] = 0
            if axs[key]["row"] == column_size - 1:
                line_end_pos[1] = 1

            if axs[key]["col"] == 0:
                line_end_pos[0] = 1
            if axs[key]["col"] == column_size - 1:
                line_end_pos[0] = 0

            infig_position = mytrans.transform(line_end_pos)

            xpos, ypos = main_ax.transLimits.transform(
                (z[chosen_point, 0], z[chosen_point, 1])
            )

            mytrans2 = main_ax.transAxes + main_ax.figure.transFigure.inverted()
            infig_position_start = mytrans2.transform([xpos, ypos])

            color = pal[key] if color_points else "k"
            lines_list.append(
                lines.Line2D(
                    [infig_position_start[0], infig_position[0]],
                    [infig_position_start[1], infig_position[1]],
                    color=color,
                    transform=fig.transFigure,
                    **line_kwargs,
                )
            )
    if draw_lines:
        for l in lines_list:
            fig.lines.append(l)

    gs.update(wspace=0, hspace=0)
    # gs.update(wspace=0.5, hspace=0.5)

    fig = plt.gcf()

    if ax is not None:
        buf = io.BytesIO()
        plt.savefig(buf, dpi=300, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        im = Image.open(buf)
        ax.imshow(im)
        plt.close(fig)

    return fig, axs, main_ax
