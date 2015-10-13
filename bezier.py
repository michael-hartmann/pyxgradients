import numpy as np
from math import exp,lgamma

# see https://pypi.python.org/pypi/colorspacious/
from colorspacious import cspace_converter

class BezierModel(object):
    def __init__(self, xp, yp):
        self._xp = list(xp)
        self._yp = list(yp)

    def get_bezier_points_at(self, at, grid=256):
        at = np.asarray(at)

        # The Bezier curve is parameterized by a value t which ranges from 0
        # to 1. However, there is a nonlinear relationship between this value
        # and arclength. We want to parameterize by t', which measures
        # normalized arclength. To do this, we have to calculate the function
        # arclength(t), and then invert it.
        t = np.linspace(0, 1, grid)

        x, y = Bezier(list(zip(self._xp, self._yp)), t).T
        x_deltas = np.diff(x)
        y_deltas = np.diff(y)

        arclength_deltas = np.empty(t.shape)
        arclength_deltas[0] = 0

        np.hypot(x_deltas, y_deltas, out=arclength_deltas[1:])
        arclength = np.cumsum(arclength_deltas)
        arclength /= arclength[-1]

        # Now (t, arclength) is a LUT describing the t -> arclength mapping
        # Invert it to get at -> t
        at_t = np.interp(at, arclength, t)

        # And finally look up at the Bezier values at at_t
        # (Might be quicker to np.interp againts x and y, but eh, doesn't
        # really matter.)
        return Bezier(list(zip(self._xp, self._yp)), at_t).T



def Bernstein(n, k):
    """Bernstein polynomial.

    """
    # binom
    coeff = exp(lgamma(1+n)-lgamma(1+k)-lgamma(1+n-k))

    return lambda x: coeff*x**k*(1-x)**(n-k)


def Bezier(points, at):
    """Build Bézier curve from points.

    """
    at = np.asarray(at)
    at_flat = at.ravel()
    N = len(points)
    curve = np.zeros((at_flat.shape[0], 2))
    for ii in range(N):
        curve += np.outer(Bernstein(N - 1, ii)(at_flat), points[ii])
    return curve.reshape(at.shape + (2,))


if __name__ == "__main__":
    # colormap viridis
    viridis = {
        "xp": [ 22.674387857633945, 11.221508276482126, -14.356589454756971, -47.18817758739222, -34.59001004812521, -6.0516291196352654 ],
        "yp": [ -20.102530541012214, -33.08246073298429, -42.24476439790574, -5.595549738219887, 42.5065445026178, 40.13395157135497 ],
        "min_Jp": 18.8671875,
        "max_Jp": 92.5
    }

    bezier = BezierModel(viridis["xp"], viridis["yp"])

    # wir wollen (R,G,B) fuer den Wert bei at
    at = 0.0784313725490196
    ap,bp = bezier.get_bezier_points_at([at])
    Jp = (viridis["max_Jp"] - viridis["min_Jp"]) * at + viridis["min_Jp"]

    # CAM02-UCS ist ein Farbraum, sRGB ist der RGB Farbraum mit Werten fuer rot, gruen, blau in [0,1]
    sRGB = cspace_converter("CAM02-UCS", "sRGB1")(np.column_stack((Jp, ap, bp)))
    print(sRGB)
