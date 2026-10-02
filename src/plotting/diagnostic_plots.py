"""
Diagnostic figures: estimated-vs-true scatters, energy-difference matching,
decay-rate comparison, schedule/population traces and the ramp-asymmetry sweep.

Styled after the reference figure in `notebooks/lzs_spectroscopy.ipynb`: black
markers for data, a red dashed "estimate = true" diagonal, frameless legends,
no grid and no gratuitous colour. Appearance comes from `plotting.style`.
"""
import numpy as np
import matplotlib.pyplot as plt

from . import style


def plot_estimate_vs_true(true_vals, est_vals, xlabel, ylabel,
                          est_vals_2=None, labels=("w/ decay fit", "w/o decay fit"),
                          ax=None, figsize=None, show=False,
                          diag_label="estimate = true", diag_loc="upper left",
                          series_loc="lower right"):
    """
    Estimated vs true scatter with the red dashed identity line.

    A second estimate series (e.g. without the decay fit) is drawn as black
    crosses, so the two are distinguished by marker rather than by colour.
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_SINGLE)
    else:
        fig = ax.figure

    true_vals = np.asarray(true_vals, dtype=float)
    est_vals = np.asarray(est_vals, dtype=float)

    s1 = ax.scatter(true_vals, est_vals, color=style.DATA,
                    marker=style.MARKERS[0], label=labels[0])
    handles = [s1]
    if est_vals_2 is not None:
        s2 = ax.scatter(true_vals, np.asarray(est_vals_2, dtype=float),
                        color=style.DATA, marker=style.MARKERS[1],
                        label=labels[1])
        handles.append(s2)

    span = np.concatenate([true_vals, est_vals])
    d = style.identity_line(ax, span, label=diag_label)

    style.style_axis(ax, xlabel=xlabel, ylabel=ylabel)

    # Two legends: the data series (boxed, out of the way) and the diagonal.
    if len(handles) > 1:
        leg_series = ax.legend(handles, list(labels[:len(handles)]),
                               frameon=True, fontsize=style.FONT_LEGEND,
                               loc=series_loc)
        ax.legend([d], [diag_label], frameon=False,
                  fontsize=style.FONT_LEGEND, loc=diag_loc)
        ax.add_artist(leg_series)
    else:
        ax.legend([d], [diag_label], frameon=False,
                  fontsize=style.FONT_LEGEND, loc=diag_loc)

    if created:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_energy_difference_match(matched_detected, matched_true,
                                 ax=None, figsize=None, show=False):
    """Detected vs true energy differences, sober version of `scatter_plot`."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_SINGLE)
    else:
        fig = ax.figure

    matched_detected = np.asarray(matched_detected, dtype=float)
    matched_true = np.asarray(matched_true, dtype=float)

    ax.scatter(matched_detected, matched_true, color=style.DATA,
               marker=style.MARKERS[0], zorder=3)
    style.identity_line(ax, np.concatenate([matched_detected, matched_true]))

    style.style_axis(ax, xlabel=r"detected $\Delta\varepsilon$",
                     ylabel=r"true $\Delta\varepsilon$",
                     legend=True, legend_loc="upper left")
    if created:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_decay_rates(lambdas, predicted_rates=None, freqs=None,
                     xlabel=None, ylabel=r"decay rate $\lambda_{mn}$",
                     ax=None, figsize=None, show=False):
    """
    Fitted decay rates as black markers, with red dashed horizontal lines at
    the predicted rates -- the third panel of the reference figure.
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_SINGLE)
    else:
        fig = ax.figure

    lambdas = np.asarray(lambdas, dtype=float)
    if freqs is not None:
        x = np.asarray(freqs, dtype=float)
        xlab = xlabel or r"$\omega_{mn} / 2\pi$"
    else:
        x = np.arange(len(lambdas))
        xlab = xlabel or "mode index"

    ax.plot(x, lambdas, style.MARKERS[0], color=style.DATA, linestyle="none")

    if predicted_rates is not None:
        for i, rate in enumerate(np.atleast_1d(predicted_rates)):
            ax.axhline(rate, color=style.REFERENCE, linestyle="--",
                       lw=style.REF_LINEWIDTH,
                       label="predicted" if i == 0 else None)
        ax.legend(frameon=False, fontsize=style.FONT_LEGEND, loc="best")

    style.style_axis(ax, xlabel=xlab, ylabel=ylabel)
    if created:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_schedule(t_list, common_schedule, hz_schedules, hx_schedules,
                  ax=None, figsize=None, show=False):
    """
    Ramp schedules. Common schedule in black, per-qubit Z in greys (solid) and
    per-qubit X in greys (dashed) -- distinguished by dash pattern and shade,
    not by colour.
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_WIDE)
    else:
        fig = ax.figure

    ax.plot(t_list, common_schedule, color=style.DATA,
            lw=style.LINEWIDTH, label=r"common ($H_0 + H_{zz}$)")

    coloured = style.PALETTE != "sober"
    n_z = len(hz_schedules)
    for i, s in enumerate(hz_schedules):
        colour = (style.CYCLE[(i + 1) % len(style.CYCLE)] if coloured
                  else str(0.30 + 0.35 * i / max(n_z - 1, 1)))
        ax.plot(t_list, s, color=colour, lw=style.LINEWIDTH,
                label=rf"$H_z[{i}]$")
    n_x = len(hx_schedules)
    for i, s in enumerate(hx_schedules):
        colour = (style.CYCLE[(i + 1) % len(style.CYCLE)] if coloured
                  else str(0.30 + 0.35 * i / max(n_x - 1, 1)))
        ax.plot(t_list, s, color=colour, lw=style.LINEWIDTH, linestyle="--",
                label=rf"$H_x[{i}]$")

    style.style_axis(ax, xlabel=r"$t$", ylabel=r"$s(t)$",
                     legend=True, legend_loc="best")
    if created:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_instantaneous_spectrum(t_list, egvals, ax=None, figsize=None,
                                show=False):
    """Instantaneous eigenvalues vs time, all in black."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_SINGLE)
    else:
        fig = ax.figure

    egvals = np.asarray(egvals, dtype=float)
    for k in range(egvals.shape[1]):
        ax.plot(t_list, egvals[:, k], color=style.DATA, lw=style.LINEWIDTH)

    style.style_axis(ax, xlabel=r"$t$", ylabel=r"$E$")
    if created:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_eigenbasis_populations(t_all, pops_all, tr=None, tw=None,
                                ax=None, figsize=None, show=False):
    """
    Eigenbasis populations over time. Levels are separated by shade and dash
    pattern; segment boundaries are thin grey vertical lines.
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_WIDE)
    else:
        fig = ax.figure

    pops_all = np.asarray(pops_all, dtype=float)
    n = pops_all.shape[1]
    dashes = ["-", "--", "-.", ":"]
    for k in range(n):
        # Colour from the active palette's cycle, but keep the dash pattern so
        # the panel still reads correctly in greyscale.
        if len(style.CYCLE) > 1 and style.PALETTE != "sober":
            colour = style.CYCLE[k % len(style.CYCLE)]
        else:
            colour = str(0.0 + 0.55 * k / max(n - 1, 1))
        ax.plot(t_all, pops_all[:, k], color=colour,
                linestyle=dashes[k % len(dashes)], lw=style.LINEWIDTH,
                label=rf"$|E_{{{k}}}\rangle$")

    if tr is not None:
        ax.axvline(tr, color=style.GUIDE, linestyle="--", lw=0.9, zorder=0)
    if tr is not None and tw is not None:
        ax.axvline(tr + tw, color=style.GUIDE, linestyle=":", lw=0.9, zorder=0)

    ax.set_ylim(-0.02, 1.02)
    style.style_axis(ax, xlabel=r"$t$", ylabel="population",
                     legend=True, legend_loc="best")
    if created:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_eigenvector_weights(vecs_mid, egvals_midpoint, n_qubits,
                             figsize=None, show=False):
    """Bitstring weights of each eigenvector at the midpoint, one row each."""
    vecs_mid = np.asarray(vecs_mid)
    dim = vecs_mid.shape[0]
    bitstrings = [format(i, f"0{n_qubits}b") for i in range(dim)]
    x = np.arange(dim)

    fig, axes = plt.subplots(dim, 1, sharex=True,
                             figsize=figsize or (max(5.5, dim), 1.5 * dim))
    if dim == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        weights = np.abs(vecs_mid[:, k]) ** 2
        ax.bar(x, weights, color=style.DATA, width=0.6)
        ax.set_ylim(0, 1)
        ax.set_ylabel(r"$|c|^2$", fontsize=style.FONT_LABEL)
        ax.tick_params(axis="both", labelsize=style.FONT_TICK)
        ax.text(0.98, 0.85, rf"$E_{{{k}}} = {egvals_midpoint[k]:.4f}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=style.FONT_LEGEND)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(bitstrings, rotation=90,
                             fontsize=style.FONT_TICK)
    axes[-1].set_xlabel("bitstring", fontsize=style.FONT_LABEL)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_asymmetry_vs_a(a_values, amp_sum_smallest3, ramp1_z_areas,
                        ax=None, figsize=None, show=False):
    """
    Summed small-peak amplitude and Z ramp1 area against the asymmetry a.

    Two quantities on different scales share the panel via a twin axis; they
    are separated by marker and dash pattern, with the secondary series in grey
    so the primary one still reads first.
    """
    a_values = np.asarray(a_values, dtype=float)
    amp_sum_smallest3 = np.asarray(amp_sum_smallest3, dtype=float)
    ramp1_z_areas = np.asarray(ramp1_z_areas, dtype=float)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_SINGLE)
    else:
        fig = ax.figure

    ln1, = ax.plot(a_values, amp_sum_smallest3, marker=style.MARKERS[0],
                   color=style.DATA, lw=style.LINEWIDTH,
                   label="sum of 3 smallest $|A|$")
    style.style_axis(ax, xlabel=r"asymmetry $a$",
                     ylabel="sum of 3 smallest amplitudes")

    ax2 = ax.twinx()
    ln2, = ax2.plot(a_values, ramp1_z_areas, marker=style.MARKERS[2],
                    linestyle="--", color=style.SECONDARY,
                    lw=style.LINEWIDTH, label="Z ramp1 area")
    ax2.set_ylabel("Z ramp1 area", fontsize=style.FONT_LABEL,
                   color=style.SECONDARY)
    ax2.tick_params(axis="y", labelsize=style.FONT_TICK,
                    labelcolor=style.SECONDARY, direction="in")
    ax2.spines["right"].set_visible(True)

    ax.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()],
              loc="upper center", frameon=False, fontsize=style.FONT_LEGEND,
              bbox_to_anchor=(0.5, 1.16), ncol=2, columnspacing=1.2)

    if created:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_amplitude_vs_area(ramp1_z_areas, amp_sum_smallest3,
                           ax=None, figsize=None, show=False):
    """Summed small-peak amplitude against the Z ramp1 area."""
    ramp1_z_areas = np.asarray(ramp1_z_areas, dtype=float)
    amp_sum_smallest3 = np.asarray(amp_sum_smallest3, dtype=float)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_SINGLE)
    else:
        fig = ax.figure

    ax.plot(ramp1_z_areas, amp_sum_smallest3, "-", color=style.SUPPORT,
            lw=0.9, zorder=0)
    ax.plot(ramp1_z_areas, amp_sum_smallest3, style.MARKERS[0],
            color=style.DATA, linestyle="none", zorder=3)
    style.style_axis(ax, xlabel="Z ramp1 area",
                     ylabel="sum of 3 smallest amplitudes")

    if created:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_asymmetry_sweep(a_values, amp_sum_smallest3, ramp1_z_areas,
                         figsize=None, show=False):
    """
    Both asymmetry-sweep panels as two separate figures, so each can be placed
    independently in the paper. Returns (fig_vs_a, fig_vs_area).
    """
    fig_a, _ = plot_asymmetry_vs_a(a_values, amp_sum_smallest3, ramp1_z_areas,
                                   figsize=figsize)
    fig_area, _ = plot_amplitude_vs_area(ramp1_z_areas, amp_sum_smallest3,
                                         figsize=figsize)
    if show:
        plt.show()
    return fig_a, fig_area
