"""
Rigid stick object representing a wire segment.

A thin BoxObject with a fixed aspect ratio. Part of the Modeling API:
generates MuJoCo XML for a rigid body with collision and visual geometry.
"""

from robosuite.models.objects.primitive.box import BoxObject

# Visual colors for distinguishing sticks in the renderer
STICK_COLORS = [
    [0.85, 0.20, 0.15, 1.0],  # red
    [0.15, 0.50, 0.85, 1.0],  # blue
    [0.15, 0.70, 0.25, 1.0],  # green
    [0.90, 0.65, 0.10, 1.0],  # yellow
    [0.70, 0.15, 0.85, 1.0],  # purple
]


class StickObject(BoxObject):
    """
    Thin elongated rigid body representing a wire segment.

    The stick lies along the x-axis. Half-extents:
        x = length / 2   (long axis)
        y = z = radius   (cross-section)

    Args:
        name: Unique MuJoCo object name.
        length: Full length in metres (default 0.18 m).
        radius: Cross-section half-extent in metres (default 0.012 m).
        color_idx: Index into STICK_COLORS for visual distinction.
    """

    def __init__(
        self,
        name: str,
        length: float = 0.18,
        radius: float = 0.012,
        color_idx: int = 0,
    ):
        # BoxObject size is half-extents: [x, y, z]
        super().__init__(
            name=name,
            size=[length / 2, radius, radius],
            rgba=STICK_COLORS[color_idx % len(STICK_COLORS)],
            density=500,
            friction=(0.3, 0.005, 0.0001),
        )
        self.length = length
        self.radius = radius
