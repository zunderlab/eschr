import collections.abc as cabc
import sys
from copy import copy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import Cycler, cycler
from matplotlib import patheffects
from matplotlib.colors import is_color_like, to_hex
from matplotlib.pyplot import get_cmap
from pandas.api.types import is_categorical_dtype

sys.setrecursionlimit(1000000)

# flake8: noqa: C411
# flake8: noqa: E266

# -------------------------------------------------------------------------------
# Colors in addition to matplotlib's colors
# -------------------------------------------------------------------------------

additional_colors = {
    "gold2": "#eec900",
    "firebrick3": "#cd2626",
    "khaki2": "#eee685",
    "slategray3": "#9fb6cd",
    "palegreen3": "#7ccd7c",
    "tomato2": "#ee5c42",
    "grey80": "#cccccc",
    "grey90": "#e5e5e5",
    "wheat4": "#8b7e66",
    "grey65": "#a6a6a6",
    "grey10": "#1a1a1a",
    "grey20": "#333333",
    "grey50": "#7f7f7f",
    "grey30": "#4d4d4d",
    "grey40": "#666666",
    "antiquewhite2": "#eedfcc",
    "grey77": "#c4c4c4",
    "snow4": "#8b8989",
    "chartreuse3": "#66cd00",
    "yellow4": "#8b8b00",
    "darkolivegreen2": "#bcee68",
    "olivedrab3": "#9acd32",
    "azure3": "#c1cdcd",
    "violetred": "#d02090",
    "mediumpurple3": "#8968cd",
    "purple4": "#551a8b",
    "seagreen4": "#2e8b57",
    "lightblue3": "#9ac0cd",
    "orchid3": "#b452cd",
    "indianred 3": "#cd5555",
    "grey60": "#999999",
    "mediumorchid1": "#e066ff",
    "plum3": "#cd96cd",
    "palevioletred3": "#cd6889",
}


def embedding(
    adata,
    color=None,
    sort_order=True,
    neighbors_key=None,
    scale_factor=None,
    cmap=None,
    palette=None,
    na_color="lightgray",
    na_in_legend=True,
    size=None,
    frameon=None,
    legend_fontsize=None,
    legend_fontweight="bold",
    legend_loc="right margin",
    legend_fontoutline=None,
    colorbar_loc="right",
    ncols=4,
    hspace=0.25,
    wspace=None,
    title=None,
    show=True,
    save=None,
    ax=None,
    return_fig=None,
    **kwargs,
):
    """
    Make scatter plot for UMAP embedding.

    Adapted from scverse/scanpy/plotting/_tools/scatterplots.py.

    Parameters
    ----------
    adata
        Annotated data matrix.
    color
        Keys for annotations of observations/cells or variables/genes, e.g.,
        `'ann1'` or `['ann1', 'ann2']`.
    sort_order
        For continuous annotations used as color parameter, plot data points
        with higher values on top of others.
    legend_loc
        Location of legend, either `'on data'`, `'right margin'` or a valid keyword
        for the `loc` parameter of :class:`~matplotlib.legend.Legend`.
    legend_fontsize
        Numeric size in pt or string describing the size.
    legend_fontweight
        Legend font weight. A numeric value in range 0-1000 or a string.
        Defaults to `'bold'` if `legend_loc == 'on data'`, otherwise to `'normal'`.
    legend_fontoutline
        Line width of the legend font outline in pt. Draws a white outline using
        the path effect :class:`~matplotlib.patheffects.withStroke`.
    colorbar_loc
        Where to place the colorbar for continous variables. If `None`, no colorbar
        is added.
    size
        Point size. If `None`, is automatically computed as 120000 / n_cells.
        Can be a sequence containing the size for each cell. The order should be
        the same as in adata.obs.
    cmap
        Color map to use for continous variables. Can be a name or a
        :class:`~matplotlib.colors.Colormap` instance (e.g. `"magma`", `"viridis"`
        or `mpl.cm.cividis`), see :func:`~matplotlib.cm.get_cmap`.
        If `None`, the value of `mpl.rcParams["image.cmap"]` is used.
        The default `color_map` can be set using :func:`~scanpy.set_figure_params`.
    palette
        Colors to use for plotting categorical annotation groups.
        The palette can be a valid :class:`~matplotlib.colors.ListedColormap` name
        (`'Set2'`, `'tab20'`, …), a :class:`~cycler.Cycler` object, a dict mapping
        categories to colors, or a sequence of colors. Colors must be valid to
        matplotlib. (see :func:`~matplotlib.colors.is_color_like`).
        If `None`, `mpl.rcParams["axes.prop_cycle"]` is used unless the categorical
        variable already has colors stored in `adata.uns["{var}_colors"]`.
        If provided, values of `adata.uns["{var}_colors"]` will be set.
    na_color
        Color to use for null or masked values. Can be anything matplotlib accepts as a
        color. Used for all points if `color=None`.
    na_in_legend
        If there are missing values, whether they get an entry in the legend. Currently
        only implemented for categorical legends.
    frameon
        Draw a frame around the scatter plot. Defaults to value set in
        :func:`~scanpy.set_figure_params`, defaults to `True`.
    title
        Provide title for panels either as string or list of strings,
        e.g. `['title1', 'title2', ...]`.
    ncols
        Number of panels per row.
    wspace
        Adjust the width of the space between multiple panels.
    hspace
        Adjust the height of the space between multiple panels.
    return_fig
        Return the matplotlib figure.
    show
        Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
    ax
        A matplotlib axes object. Only works if plotting a single component.

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.
    """
    #####################
    # Argument handling #
    #####################

    basis_values = adata.obsm["X_umap"]
    # dimensions = _components_to_dimensions(
    #    components, dimensions, projection=projection, total_dims=basis_values.shape[1]
    # )
    dimensions = [(0, 1)]

    cmap = copy(get_cmap(cmap))
    cmap.set_bad(na_color)
    kwargs["cmap"] = cmap
    # Prevents warnings during legend creation
    na_color = to_hex(na_color, keep_alpha=True)

    # Vectorized arguments

    # turn color into a python list
    color = [color] if isinstance(color, str) or color is None else list(color)
    if title is not None:
        # turn title into a python list if not None
        title = [title] if isinstance(title, str) else list(title)

    # Size
    if "s" in kwargs and size is None:
        size = kwargs.pop("s")
    if size is not None:
        # check if size is any type of sequence, and if so
        # set as ndarray
        if (
            size is not None
            and isinstance(size, (cabc.Sequence, pd.Series, np.ndarray))
            and len(size) == adata.shape[0]
        ):
            size = np.array(size, dtype=float)
    else:
        size = 10000 / adata.shape[0]

    ##########
    # Layout #
    ##########
    # Most of the code is for the case when multiple plots are required

    # First set figsize based on number of subplots
    # plt.rcParams["figure.figsize"] = [3 * ncols, (3 * (len(color) / ncols))]
    plt.rcParams["figure.figsize"] = [3, 3]

    if wspace is None:
        #  try to set a wspace that is not too large or too small given the
        #  current figure size
        wspace = 0.75 / plt.rcParams["figure.figsize"][0] + 0.02

    color, dimensions = _broadcast_args(color, dimensions)

    if (
        not isinstance(color, str)
        and isinstance(color, cabc.Sequence)
        and len(color) > 1
    ) or len(dimensions) > 1:
        if ax is not None:
            raise ValueError(
                "Cannot specify `ax` when plotting multiple panels "
                "(each for a given value of 'color')."
            )

        # each plot needs to be its own panel
        fig, grid = _panel_grid(hspace, wspace, ncols, len(color))
    else:
        grid = None
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

    ############
    # Plotting #
    ############
    axs = []

    # use itertools.product to make a plot for each color and for each component
    # For example if color=[gene1, gene2] and components=['1,2, '2,3'].
    # The plots are: [
    #     color=gene1, components=[1,2], color=gene1, components=[2,3],
    #     color=gene2, components = [1, 2], color=gene2, components=[2,3],
    # ]
    for count, (value_to_plot, dims) in enumerate(zip(color, dimensions)):
        color_source_vector = _get_color_source_vector(adata, value_to_plot)

        if is_categorical_dtype(color_source_vector):
            categorical = True
        else:
            categorical = False
        if categorical:
            if palette:
                _set_colors_for_categorical_obs(adata, value_to_plot, palette)
            else:
                _set_default_colors_for_categorical_obs(adata, value_to_plot)
            values = pd.Categorical(adata.obs[value_to_plot])
            cmap = dict(zip(values.categories, adata.uns[value_to_plot + "_colors"]))
            color_vector = pd.Categorical(adata.obs[value_to_plot].map(cmap))
        else:
            color_vector = color_source_vector

        # Order points
        order = slice(None)
        if sort_order is True and value_to_plot is not None and categorical is False:
            # Higher values plotted on top, null values on bottom
            order = np.argsort(-color_vector, kind="stable")[::-1]
        elif sort_order and categorical:
            # Null points go on bottom
            order = np.argsort(~pd.isnull(color_source_vector), kind="stable")
        # Set orders
        if isinstance(size, np.ndarray):
            size = np.array(size)[order]
        color_source_vector = color_source_vector[order]
        color_vector = color_vector[order]
        # print("dims")
        # print(dims)
        # print("order")
        # print(order)
        coords = basis_values[:, dims][order, :]

        # if plotting multiple panels, get the ax from the grid spec
        # else use the ax value (either user given or created previously)
        if grid:
            ax = plt.subplot(grid[count])
            axs.append(ax)
        if not frameon:
            ax.axis("off")
        if title is None:
            if value_to_plot is not None:
                ax.set_title(value_to_plot)
            else:
                ax.set_title("")
        else:
            try:
                ax.set_title(title[count])
            except IndexError:
                # ogg.warning(
                print(
                    "The title list is shorter than the number of panels. Using 'color' value instead for some plots."
                )
                ax.set_title(value_to_plot)

        # make the scatter plot
        scatter = (
            partial(ax.scatter, s=size, plotnonfinite=True)
            # if scale_factor is None
            # else partial(circles, s=size, ax=ax, scale_factor=scale_factor)  # size in circles is radius
        )

        ## add option for setting min/max bounds later
        normalize = None
        cax = scatter(
            coords[:, 0],
            coords[:, 1],
            marker=".",
            c=color_vector,
            # rasterized=settings._vector_friendly,
            norm=normalize,
            **kwargs,
        )

        # remove y and x ticks
        ax.set_yticks([])
        ax.set_xticks([])

        # set default axis_labels
        name = "UMAP"
        axis_labels = [name + str(d + 1) for d in dims]

        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.autoscale_view()

        if value_to_plot is None:
            # if only dots were plotted without an associated value
            # there is not need to plot a legend or a colorbar
            continue

        if legend_fontoutline is not None:
            path_effect = [
                patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")
            ]
        else:
            path_effect = None

        # Adding legends
        if categorical:
            _add_categorical_legend(
                ax,
                color_source_vector,
                palette=_get_palette(adata, value_to_plot),
                scatter_array=coords,
                legend_loc=legend_loc,
                legend_fontweight=legend_fontweight,
                legend_fontsize=legend_fontsize,
                legend_fontoutline=path_effect,
                na_color=na_color,
                na_in_legend=na_in_legend,
                multi_panel=bool(grid),
            )
        elif colorbar_loc is not None:
            plt.colorbar(
                cax, ax=ax, pad=0.01, fraction=0.08, aspect=30, location=colorbar_loc
            )
    axs = axs if grid else ax
    # print(show)
    if show is True:
        plt.show()
    elif show is False:
        return axs
    if return_fig is True:
        return fig


def _get_color_source_vector(
    adata, value_to_plot, use_raw=False, gene_symbols=None, layer=None, groups=None
):
    """Get array from adata that colors will be based on."""
    if value_to_plot is None:
        # Points will be plotted with `na_color`. Ideally this would work
        # with the "bad color" in a color map but that throws a warning. Instead
        # _color_vector handles this.
        # https://github.com/matplotlib/matplotlib/issues/18294
        return np.broadcast_to(np.nan, adata.n_obs)
    if (
        gene_symbols is not None
        and value_to_plot not in adata.obs.columns
        and value_to_plot not in adata.var_names
    ):
        # We should probably just make an index for this, and share it over runs
        value_to_plot = adata.var.index[adata.var[gene_symbols] == value_to_plot][
            0
        ]  # TODO: Throw helpful error if this doesn't work
    if use_raw and value_to_plot not in adata.obs.columns:
        values = adata.raw.obs_vector(value_to_plot)
    else:
        values = adata.obs_vector(value_to_plot, layer=layer)
    if groups and is_categorical_dtype(values):
        values = values.replace(values.categories.difference(groups), np.nan)
    return values


def _set_colors_for_categorical_obs(adata, value_to_plot, palette):
    """
    Sets the adata.uns[value_to_plot + '_colors'] according to the given palette.

    Parameters
    ----------
    adata
        annData object
    value_to_plot
        name of a valid categorical observation
    palette
        Palette should be either a valid :func:`~matplotlib.pyplot.colormaps` string,
        a sequence of colors (in a format that can be understood by matplotlib,
        eg. RGB, RGBS, hex, or a cycler object with key='color'

    Returns
    -------
    None
    """
    from matplotlib.colors import to_hex

    categories = adata.obs[value_to_plot].cat.categories
    # check is palette is a valid matplotlib colormap
    if isinstance(palette, str) and palette in plt.colormaps():
        # this creates a palette from a colormap. E.g. 'Accent, Dark2, tab20'
        cmap = plt.get_cmap(palette)
        colors_list = [to_hex(x) for x in cmap(np.linspace(0, 1, len(categories)))]
    elif isinstance(palette, cabc.Mapping):
        colors_list = [to_hex(palette[k], keep_alpha=True) for k in categories]
    else:
        # check if palette is a list and convert it to a cycler, thus
        # it doesnt matter if the list is shorter than the categories length:
        if isinstance(palette, cabc.Sequence):
            if len(palette) < len(categories):
                # logg.warning(
                print(
                    "Length of palette colors is smaller than the number of "
                    f"categories (palette length: {len(palette)}, "
                    f"categories length: {len(categories)}. "
                    "Some categories will have the same color."
                )
            # check that colors are valid
            _color_list = []
            for color in palette:
                if not is_color_like(color):
                    # check if the color is a valid R color and translate it
                    # to a valid hex color value
                    if color in additional_colors:
                        color = additional_colors[color]
                    else:
                        raise ValueError(
                            "The following color value of the given palette "
                            f"is not valid: {color}"
                        )
                _color_list.append(color)

            palette = cycler(color=_color_list)
        if not isinstance(palette, Cycler):
            raise ValueError(
                "Please check that the value of 'palette' is a valid "
                "matplotlib colormap string (eg. Set2), a  list of color names "
                "or a cycler with a 'color' key."
            )
        if "color" not in palette.keys:
            raise ValueError("Please set the palette key 'color'.")

        cc = palette()
        colors_list = [to_hex(next(cc)["color"]) for x in range(len(categories))]

    adata.uns[value_to_plot + "_colors"] = colors_list


def _set_default_colors_for_categorical_obs(adata, value_to_plot):
    """
    Sets the adata.uns[value_to_plot + '_colors'] using default color palettes

    Parameters
    ----------
    adata
        AnnData object
    value_to_plot
        Name of a valid categorical observation

    Returns
    -------
    None
    """
    categories = adata.obs[value_to_plot].cat.categories
    length = len(categories)

    # check if default matplotlib palette has enough colors
    if len(plt.rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
        cc = plt.rcParams["axes.prop_cycle"]()
        palette = [next(cc)["color"] for _ in range(length)]

    else:
        if length <= 20:
            palette = palettes.default_20
        elif length <= 28:
            palette = palettes.default_28
        elif length <= len(palettes.default_102):  # 103 colors
            palette = palettes.default_102
        else:
            palette = ["grey" for _ in range(length)]
            # logg.info(
            print(
                f"the obs value {value_to_plot!r} has more than 103 categories. Uniform "
                "'grey' color will be used for all categories."
            )

    _set_colors_for_categorical_obs(adata, value_to_plot, palette[:length])


def _validate_palette(adata, key):
    """
    Validate palette.

    Checks if the list of colors in adata.uns[f'{key}_colors'] is valid
    and updates the color list in adata.uns[f'{key}_colors'] if needed.
    Not only valid matplotlib colors are checked but also if the color name
    is a valid R color name, in which case it will be translated to a valid name
    """
    _palette = []
    color_key = f"{key}_colors"

    for color in adata.uns[color_key]:
        if not is_color_like(color):
            # check if the color is a valid R color and translate it
            # to a valid hex color value
            if color in additional_colors:
                color = additional_colors[color]
            else:
                # logg.warning(
                print(
                    f"The following color value found in adata.uns['{key}_colors'] "
                    f"is not valid: '{color}'. Default colors will be used instead."
                )
                _set_default_colors_for_categorical_obs(adata, key)
                _palette = None
                break
        _palette.append(color)
    # Don't modify if nothing changed
    if _palette is not None and list(_palette) != list(adata.uns[color_key]):
        adata.uns[color_key] = _palette


def _get_palette(adata, values_key: str, palette=None):
    color_key = f"{values_key}_colors"
    values = pd.Categorical(adata.obs[values_key])
    if palette:
        _set_colors_for_categorical_obs(adata, values_key, palette)
    elif color_key not in adata.uns or len(adata.uns[color_key]) < len(
        values.categories
    ):
        #  set a default palette in case that no colors or few colors are found
        _set_default_colors_for_categorical_obs(adata, values_key)
    else:
        _validate_palette(adata, values_key)
    return dict(zip(values.categories, adata.uns[color_key]))


def _broadcast_args(*args):
    """Broadcasts arguments to a common length."""
    lens = [len(arg) for arg in args]
    longest = max(lens)
    if not (set(lens) == {1, longest} or set(lens) == {longest}):
        raise ValueError(f"Could not broadast together arguments with shapes: {lens}.")
    return list(
        [[arg[0] for _ in range(longest)] if len(arg) == 1 else arg for arg in args]
    )


def _add_categorical_legend(
    ax,
    color_source_vector,
    palette: dict,
    legend_loc: str,
    legend_fontweight,
    legend_fontsize,
    legend_fontoutline,
    multi_panel,
    na_color,
    na_in_legend: bool,
    scatter_array=None,
):
    """Add a legend to the passed Axes."""
    if na_in_legend and pd.isnull(color_source_vector).any():
        if "NA" in color_source_vector:
            raise NotImplementedError(
                "No fallback for null labels has been defined if NA already in categories."
            )
        color_source_vector = color_source_vector.add_categories("NA").fillna("NA")
        palette = palette.copy()
        palette["NA"] = na_color
    cats = color_source_vector.categories

    if multi_panel is True:
        # Shrink current axis by 10% to fit legend and match
        # size of plots that are not categorical
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.91, box.height])

    if legend_loc == "right margin":
        for label in cats:
            ax.scatter([], [], c=palette[label], label=label)
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=(1 if len(cats) <= 14 else 2 if len(cats) <= 30 else 3),
            fontsize=legend_fontsize,
        )
    elif legend_loc == "on data":
        # identify centroids to put labels

        all_pos = (
            pd.DataFrame(scatter_array, columns=["x", "y"])
            .groupby(color_source_vector, observed=True)
            .median()
            # Have to sort_index since if observed=True and categorical is unordered
            # the order of values in .index is undefined. Related issue:
            # https://github.com/pandas-dev/pandas/issues/25167
            .sort_index()
        )

        for label, x_pos, y_pos in all_pos.itertuples():
            ax.text(
                x_pos,
                y_pos,
                label,
                weight=legend_fontweight,
                verticalalignment="center",
                horizontalalignment="center",
                fontsize=legend_fontsize,
                path_effects=legend_fontoutline,
            )


def _panel_grid(hspace, wspace, ncols, num_panels):
    from matplotlib import gridspec

    n_panels_x = min(ncols, num_panels)
    n_panels_y = np.ceil(num_panels / n_panels_x).astype(int)
    # each panel will have the size of rcParams['figure.figsize']
    fig = plt.figure(
        figsize=(
            n_panels_x * plt.rcParams["figure.figsize"][0] * (1 + wspace),
            n_panels_y * plt.rcParams["figure.figsize"][1],
        ),
    )
    left = 0.2 / n_panels_x
    bottom = 0.13 / n_panels_y
    gs = gridspec.GridSpec(
        nrows=n_panels_y,
        ncols=n_panels_x,
        left=left,
        right=1 - (n_panels_x - 1) * left - 0.01 / n_panels_x,
        bottom=bottom,
        top=1 - (n_panels_y - 1) * bottom - 0.1 / n_panels_y,
        hspace=hspace,
        wspace=wspace,
    )
    return fig, gs
