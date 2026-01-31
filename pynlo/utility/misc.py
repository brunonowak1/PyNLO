# -*- coding: utf-8 -*-
"""
Miscellaneous helper classes and functions.

for converting a list of png files to a .gif, considering using imagemagick:
    convert -dispose previous Background -delay 10 *.png ~/Downloads/mygif.gif
"""

__all__ = ["replace"]


# %% Imports

from unicodedata import normalize
import numpy as np
import matplotlib.pyplot as plt
import pynlo
import scipy.constants as sc
from pynlo.utility import blit
import time

# %% Helper Functions


def replace(array, values, key):
    """Copy `array` with elements given by `key` replaced by `values`."""
    array = array.copy()
    array[key] = values
    return array


# %% Array Properties for Classes


class ArrayWrapper(np.lib.mixins.NDArrayOperatorsMixin):
    """Emulates an NDArray using custom item getters and item setters."""
    def __init__(self, getter=None, setter=None):
        self._getter = getter
        self._setter = setter

    def __getitem__(self, key):
        # Return a view of the array descriptor
        def item_getter(new_key):
            return self._getter(key)[new_key]
        def item_setter(new_key, value):
            self._setter(key, replace(self._getter(key), value, new_key))
        return ArrayWrapper(getter=item_getter, setter=item_setter)

    def __setitem__(self, key, value):
        self._setter(key, value)

    def __array__(self, dtype=None):
        # Return the NumPy array
        array = self._getter(...)
        if dtype is None:
            return array
        else:
            return array.astype(dtype=dtype)

    def __repr__(self):
        return repr(self.__array__())

    def __len__(self):
        return len(self.__array__())

    def __copy__(self):
        return self.__array__()

    def __deepcopy__(self, memo):
        return self.__array__().copy()

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        Implemented to support use of the `out` ufunc keyword and seamless
        conversion to NumPy arrays.

        Modified from NumPy docs, "__array_ufunc__ for ufuncs"

        """
        #---- Convert Input to NumPy Arrays
        inputs = tuple(x.__array__() if isinstance(x, ArrayWrapper) else x
                       for x in inputs)

        #---- Apply Ufunc
        if out:
            # Convert Output to NumPy Arrays
            outputs = []
            out_args = []
            for idx, output in enumerate(out):
                if isinstance(output, ArrayWrapper):
                    outputs.append([idx, output])
                    out_args.append(output.__array__())
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)

            # Apply Ufunc
            result = getattr(ufunc, method)(*inputs, **kwargs)

            # Write In-Place Output to ArrayWrapper
            for idx, output in outputs:
                output[...] = out_args[idx] # "in place" equivalent
        else:
            result = getattr(ufunc, method)(*inputs, **kwargs)

        #---- Return Result
        if method == 'at':
            return None # no return value
        else:
            return result

    def __getattr__(self, attr):
        """Catch-all for other NumPy functions"""
        return getattr(self.__array__(), attr)


class SettableArrayProperty(property):
    """
    A subclass of `property` that allows extending the getter and setter
    formalism to NumPy array elements.

    Notes
    -----
    To allow usage of both `__get__`/`__getitem__` and `__set__`/`__setitem__`,
    the methods fed into `SettableArrayProperty` must contain a keyword argument and logic
    for processing the keys used by `__getitem__` and `__setitem__`. In the
    `setter` method, the `value` parameter must precede the `key` parameter. In
    the following example, the default key is an open slice (ellipsis), the
    entire array is retrieved when individual elements are not requested.::

        class C(object):
            def __init__(self):
                self.x = np.array([1,2,3,4])

            @SettableArrayProperty
            def y(self, key=...):
                return self.x[key]**2

            @y.setter
            def y(self, value, key=...):
                self.x[key] = value**0.5

    See the documentation of `property` for other implementation details.

    """
    def __get__(self, obj, objtype):
        # Return self if not instantiated
        if obj is None:
            return self

        # Define Item Getter and Setter
        def item_getter(key):
            return self.fget(obj, key)

        def item_setter(key, value):
            if self.fset is None:
                self.__set__(obj, value) # raise AttributeError if fset is None
            self.fset(obj, value, key)

        # Return ndarray with custom item getters and item setters
        array = ArrayWrapper(getter=item_getter, setter=item_setter)
        return array

def plot_results(pulse_out, model, z, a_t, a_v, plot="frq", num="Simulation Results", 
                scale="db",db_floor=-40.0,normalize=True):
    """
    plot PyNLO simulation results

    scale:
        "db"   -> dB scale (normalized so max = 0 dB)
        "lin"  -> linear power (optionally normalized so max = 1)
        "both" -> make two figures: dB and linear

    Args:
        pulse_out (object):
            pulse instance that is used for the time and frequency grid
            (so actually could also be input pulse)
        z (1D array): simulation z grid points
        a_t (2D array): a_t at each z grid point
        a_v (2D array): a_v at each z grid point
        plot (string, optional):
            whether to plot the frequency domain with frequency or wavelength
            on the x axis, default is frequency
    """
    assert plot in ("frq", "wvl"), "plot must be 'frq' or 'wvl'"
    assert scale in ("db", "lin", "both"), "scale must be 'db', 'lin', or 'both'"

    # ---- if both, just call twice and return ----
    if scale == "both":
        plot_results(pulse_out, model, z, a_t, a_v, plot=plot, num=f"{num} (dB)",
                     scale="db", db_floor=db_floor, normalize=normalize)
        return plot_results(pulse_out, model, z, a_t, a_v, plot=plot, num=f"{num} (linear)",
                            scale="lin", db_floor=db_floor, normalize=normalize)

    # ---- local helpers (so db_floor/normalize are in scope) ----
    def _norm(x):
        x = np.asarray(x)
        if normalize:
            x = x / (np.max(x) + 1e-30)
        return x

    def _as_db(x):
        x = _norm(x)
        y = 10.0 * np.log10(x + 1e-30)
        if db_floor is not None:
            y = np.maximum(y, db_floor)
        return y

    def _as_lin(x):
        return _norm(x)

    # ---- layout ----
    fig = plt.figure(num=num, clear=True)
    ax0 = plt.subplot2grid((3, 2), (0, 0), rowspan=1)
    ax1 = plt.subplot2grid((3, 2), (0, 1), rowspan=1)
    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, sharex=ax0)
    ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2, sharex=ax1)

    # ---- build linear powers first ----
    p_v = np.abs(a_v) ** 2                     # Nz x Nv (linear)
    p_t = np.abs(a_t) ** 2                     # Nz x Nt (linear)
    p_wl = p_v * model.dv_dl                   # Nz x Nv (linear "per wavelength")

    # ---- choose which scale arrays to plot ----
    if scale == "db":
        p_v_plot  = _as_db(p_v)
        p_t_plot  = _as_db(p_t)
        p_wl_plot = _as_db(p_wl)
        ylab = "Power (dB)"
        vmin, vmax = db_floor, 0.0
        # keep same y-limits as before
        yline_lim = (-50, 10)
    else:  # "lin"
        p_v_plot  = _as_lin(p_v)
        p_t_plot  = _as_lin(p_t)
        p_wl_plot = _as_lin(p_wl)
        ylab = "Power (norm.)" if normalize else "Power (linear)"
        vmin, vmax = (0.0, 1.0) if normalize else (None, None)
        yline_lim = (0.0, 1.05) if normalize else (None, None)

    # ---- frequency / wavelength plots ----
    if plot == "frq":
        x = 1e-12 * pulse_out.v_grid  # THz
        ax0.plot(x, p_v_plot[0],  color="b")
        ax0.plot(x, p_v_plot[-1], color="g")
        ax2.pcolorfast(
            x,
            1e3 * z,
            p_v_plot[:-1, :-1],
            vmin=vmin,
            vmax=vmax,
            cmap="CMRmap_r_t",
        )
        ax2.set_xlabel("Frequency (THz)")

    else:  # "wvl"
        wl_grid = sc.c / pulse_out.v_grid          # meters
        x = 1e6 * wl_grid                          # um
        ax0.plot(x, p_wl_plot[0],  color="b")
        ax0.plot(x, p_wl_plot[-1], color="g")
        ax2.pcolorfast(
            x,
            1e3 * z,
            p_wl_plot[:-1, :-1],
            vmin=vmin,
            vmax=vmax,
            cmap="CMRmap_r_t",
        )
        ax2.invert_xaxis()
        ax2.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")

    # ---- time plots ----
    xt = 1e12 * pulse_out.t_grid  # ps
    ax1.plot(xt, p_t_plot[0],  color="b")
    ax1.plot(xt, p_t_plot[-1], color="g")
    ax3.pcolorfast(
        xt,
        1e3 * z,
        p_t_plot[:-1, :-1],
        vmin=vmin,
        vmax=vmax,
        cmap="CMRmap_r_t",
    )
    ax3.set_xlabel("Time (ps)")

    # ---- labels/limits ----
    ax0.set_ylabel(ylab)
    ax2.set_ylabel("Propagation Distance (mm)")

    if yline_lim[0] is not None:
        ax0.set_ylim(*yline_lim)
        ax1.set_ylim(*yline_lim)
    else:
        # if not normalized linear, leave autoscale
        pass

    fig.tight_layout()
    fig.show()
    return fig, np.array([[ax0, ax1], [ax2, ax3]])


def animate(
    pulse_out, model, z, a_t, a_v, plot="frq", save=False, p_ref=None, figsize=None
):
    """
    replay the real time simulation

    Args:
        pulse_out (object):
            reference pulse instance for time and frequency grid
        model (object):
            pynlo.model.UPE or NLSE instance used in the simulation
        z (1D array):
            z grid returned from the call to model.simulate()
        a_t (2D array):
            time domain electric fields returned from the call to
            model.simulate()
        a_v (TYPE):
            frequency domain electric fields returned from the call to
            model.simulate()
        plot (str, optional):
            "frq", "wvl" or "time"
        save (bool, optional):
            save figures to fig/ folder, default is False (see ezgif.com)
        p_ref (pulse instance, optional):
            a reference pulse to overlay all the plots, useful if you have a
            measured spectrum to compare against to
        figsize: (matplotlib figure size, optional)
    """
    assert np.any(
        [plot == "frq", plot == "wvl", plot == "time"]
    ), "plot must be 'frq' or 'wvl'"
    assert isinstance(pulse_out, pynlo.light.Pulse)
    assert isinstance(model, (pynlo.model.UPE, pynlo.model.NLSE))
    assert isinstance(p_ref, pynlo.light.Pulse) or p_ref is None
    pulse_out: pynlo.light.Pulse
    model: pynlo.model.UPE
    p_ref: pynlo.light.Pulse

    # fig, ax = plt.subplots(2, 1, num="Replay of Simulation", clear=True)
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    ax0, ax1 = ax

    wl_grid = sc.c / pulse_out.v_grid

    p_v = abs(a_v) ** 2
    p_t = abs(a_t) ** 2
    phi_t = np.angle(a_t)
    phi_v = np.angle(a_v)

    vg_t = pulse_out.v_ref + np.gradient(
        np.unwrap(phi_t) / (2 * np.pi), pulse_out.t_grid, edge_order=2, axis=1
    )
    tg_v = pulse_out.t_ref - np.gradient(
        np.unwrap(phi_v) / (2 * np.pi), pulse_out.v_grid, edge_order=2, axis=1
    )

    initialized = False
    for n in range(len(a_t)):
        # [i.clear() for i in [ax0, ax1]]

        if plot == "time":
            if not initialized:
                (l0,) = ax0.semilogy(pulse_out.t_grid * 1e12, p_t[n], ".", markersize=1)
                (l1,) = ax1.plot(
                    pulse_out.t_grid * 1e12,
                    vg_t[n] * 1e-12,
                    ".",
                    markersize=1,
                    # label=f"z = {np.round(z[n] * 1e3, 3)} mm",
                )

                ax0.set_title("Instantaneous Power")
                ax0.set_ylabel("J / s")
                ax0.set_xlabel("Delay (ps)")
                ax1.set_ylabel("Frequency (THz)")
                ax1.set_xlabel("Delay (ps)")

                excess = 0.05 * (pulse_out.v_grid.max() - pulse_out.v_grid.min())
                ax0.set_ylim(top=max(p_t[n] * 1e1), bottom=max(p_t[n] * 1e-9))
                ax1.set_ylim(
                    top=1e-12 * (pulse_out.v_grid.max() + excess),
                    bottom=1e-12 * (pulse_out.v_grid.min() - excess),
                )

                fr_number = ax1.annotate(
                    "0",
                    (0, 1),
                    xycoords="axes fraction",
                    xytext=(10, -10),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    animated=True,
                )
                fr_number.set_text(f"z = {np.round(z[n] * 1e3, 3)} mm")

                fig.tight_layout()

                bm = blit.BlitManager(fig.canvas, [l0, l1, fr_number])
                bm.update()
                initialized = True

            else:
                l0.set_ydata(p_t[n])
                l1.set_ydata(vg_t[n] * 1e-12)
                excess = 0.05 * (pulse_out.v_grid.max() - pulse_out.v_grid.min())
                ax0.set_ylim(top=max(p_t[n] * 1e1), bottom=max(p_t[n] * 1e-9))
                ax1.set_ylim(
                    top=1e-12 * (pulse_out.v_grid.max() + excess),
                    bottom=1e-12 * (pulse_out.v_grid.min() - excess),
                )
                fr_number.set_text(f"z = {np.round(z[n] * 1e3, 3)} mm")

                bm.update()

        if plot == "frq":
            if not initialized:
                (l0,) = ax0.semilogy(
                    pulse_out.v_grid * 1e-12, p_v[n], ".", markersize=1
                )
                (l1,) = ax1.plot(
                    pulse_out.v_grid * 1e-12,
                    tg_v[n] * 1e12,
                    ".",
                    markersize=1,
                    # label=f"z = {np.round(z[n] * 1e3, 3)} mm",
                )

                if p_ref is not None:
                    ax0.semilogy(p_ref.v_grid * 1e-12, p_ref.p_v, ".", markersize=1)

                ax0.set_title("Power Spectrum")
                ax0.set_ylabel("J / Hz")
                ax0.set_xlabel("Frequency (THz)")
                ax1.set_ylabel("Delay (ps)")
                ax1.set_xlabel("Frequency (THz)")

                excess = 0.05 * (pulse_out.t_grid.max() - pulse_out.t_grid.min())
                ax0.set_ylim(top=max(p_v[n] * 1e1), bottom=max(p_v[n] * 1e-9))
                ax1.set_ylim(
                    top=1e12 * (pulse_out.t_grid.max() + excess),
                    bottom=1e12 * (pulse_out.t_grid.min() - excess),
                )

                fr_number = ax1.annotate(
                    "0",
                    (0, 1),
                    xycoords="axes fraction",
                    xytext=(10, -10),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    animated=True,
                )
                fr_number.set_text(f"z = {np.round(z[n] * 1e3, 3)} mm")

                fig.tight_layout()

                bm = blit.BlitManager(fig.canvas, [l0, l1, fr_number])
                bm.update()
                initialized = True

            else:
                l0.set_ydata(p_v[n])
                l1.set_ydata(tg_v[n] * 1e12)
                excess = 0.05 * (pulse_out.t_grid.max() - pulse_out.t_grid.min())
                ax0.set_ylim(top=max(p_v[n] * 1e1), bottom=max(p_v[n] * 1e-9))
                ax1.set_ylim(
                    top=1e12 * (pulse_out.t_grid.max() + excess),
                    bottom=1e12 * (pulse_out.t_grid.min() - excess),
                )
                fr_number.set_text(f"z = {np.round(z[n] * 1e3, 3)} mm")

                bm.update()

        if plot == "wvl":
            if not initialized:
                (l0,) = ax0.semilogy(
                    wl_grid * 1e6, p_v[n] * model.dv_dl, ".", markersize=1
                )
                (l1,) = ax1.plot(
                    wl_grid * 1e6,
                    tg_v[n] * 1e12,
                    ".",
                    markersize=1,
                    # label=f"z = {np.round(z[n] * 1e3, 3)} mm",
                )

                if p_ref is not None:
                    ax0.semilogy(
                        p_ref.wl_grid * 1e6, p_ref.p_v * model.dv_dl, ".", markersize=1
                    )

                ax0.set_title("Power Spectrum")
                ax0.set_ylabel("J / m")
                ax0.set_xlabel("Wavelength ($\\mathrm{\\mu m}$)")
                ax1.set_ylabel("Delay (ps)")
                ax1.set_xlabel("Wavelength ($\\mathrm{\\mu m}$)")

                excess = 0.05 * (pulse_out.t_grid.max() - pulse_out.t_grid.min())
                ax0.set_ylim(
                    top=max(p_v[n] * model.dv_dl * 1e1),
                    bottom=max(p_v[n] * model.dv_dl * 1e-9),
                )
                ax1.set_ylim(
                    top=1e12 * (pulse_out.t_grid.max() + excess),
                    bottom=1e12 * (pulse_out.t_grid.min() - excess),
                )

                fr_number = ax1.annotate(
                    "0",
                    (0, 1),
                    xycoords="axes fraction",
                    xytext=(10, -10),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    animated=True,
                )
                fr_number.set_text(f"z = {np.round(z[n] * 1e3, 3)} mm")

                fig.tight_layout()

                bm = blit.BlitManager(fig.canvas, [l0, l1, fr_number])
                bm.update()
                initialized = True

            else:
                l0.set_ydata(p_v[n] * model.dv_dl)
                l1.set_ydata(tg_v[n] * 1e12)
                excess = 0.05 * (pulse_out.t_grid.max() - pulse_out.t_grid.min())
                ax0.set_ylim(
                    top=max(p_v[n] * model.dv_dl * 1e1),
                    bottom=max(p_v[n] * model.dv_dl * 1e-9),
                )
                ax1.set_ylim(
                    top=1e12 * (pulse_out.t_grid.max() + excess),
                    bottom=1e12 * (pulse_out.t_grid.min() - excess),
                )
                fr_number.set_text(f"z = {np.round(z[n] * 1e3, 3)} mm")

                bm.update()

        # ax1.legend(loc="upper center")
        # if n == 0:
        #     fig.tight_layout()

        if save:
            s_max = str(len(a_t) - 1)
            s = str(n)
            s = "0" * (len(s_max) - len(s)) + s
            s = "fig/" + s + ".png"
            plt.savefig(s, transparent=True, dpi=300)
        else:
            time.sleep(0.01)


def package_sim_output(simulate):
    def wrapper(self, *args, **kwargs):
        pulse_out, z, a_t, a_v = simulate(self, *args, **kwargs)
        model = self

        class result:
            def __init__(self):
                self.pulse_out = pulse_out.copy()
                self.z = z
                self.a_t = a_t
                self.a_v = a_v
                self.p_t = abs(a_t) ** 2
                self.p_v = abs(a_v) ** 2
                self.phi_v = np.angle(a_v)
                self.phi_t = np.angle(a_t)
                self.model = model

            def animate(self, plot, save=False, p_ref=None, figsize=None):
                animate(
                    self.pulse_out,
                    self.model,
                    self.z,
                    self.a_t,
                    self.a_v,
                    plot=plot,
                    save=save,
                    p_ref=p_ref,
                    figsize=figsize,
                )

            def plot(self, plot, num="Simulation Results", scale="db", db_floor=-40.0, normalize=True):
                return plot_results(
                    self.pulse_out,
                    self.model,
                    self.z,
                    self.a_t,
                    self.a_v,
                    plot=plot,
                    num=num,
                    scale=scale,
                    db_floor=db_floor,
                    normalize=normalize,
                )

            def save(self, path, filename):
                assert path != "" and isinstance(path, str), "give a save path"
                assert filename != "" and isinstance(filename, str)

                path = path + "/" if path[-1] != "" else path
                np.save(path + filename + "_t_grid.npy", self.pulse_out.t_grid)
                np.save(path + filename + "_v_grid.npy", self.pulse_out.v_grid)
                np.save(path + filename + "_z.npy", self.z)
                np.save(path + filename + "_amp_t.npy", abs(self.pulse_out.a_t))
                np.save(path + filename + "_amp_v.npy", abs(self.pulse_out.a_v))
                np.save(path + filename + "_phi_t.npy", np.angle(self.pulse_out.a_t))
                np.save(path + filename + "_phi_v.npy", np.angle(self.pulse_out.a_v))

        return result()

    return wrapper
