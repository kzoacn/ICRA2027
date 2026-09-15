"""Use diagonal room in a measured caddy opening without pitching a held book."""
import numpy as np


def compartment_yaw_rotation(source, destination, preferred_delta, *, alternate_sources=()):
    """Keep height unchanged and use diagonal space in a rectangular opening.

    The opening is already a measured inner compartment. Retain 2 mm on
    each side and choose the smallest additional yaw that fits the payload.
    If no yaw fits, preserve the original pose rather than pitching the hand.
    """
    capacity = np.asarray(destination.extents_m) - .004

    def fits(delta):
        for hypothesis in (source, *alternate_sources):
            extent = np.abs(destination.axes_world.T @ delta @ hypothesis.axes_world) @ hypothesis.extents_m
            if not np.all(extent <= capacity):
                return False
        return True

    if fits(preferred_delta):
        return preferred_delta.copy()
    for degrees in range(1, 91):
        for sign in (1., -1.):
            angle = np.deg2rad(degrees * sign)
            c, s = np.cos(angle), np.sin(angle)
            yaw = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))
            delta = yaw @ preferred_delta
            if fits(delta):
                return delta
    return preferred_delta.copy()


