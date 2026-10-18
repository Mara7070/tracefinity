"""Pure canonical geometry for one closed 2D outline.

Coordinates retain their input frame and units. Counter-clockwise winding uses
the mathematical signed-area convention; no image-coordinate Y flip is applied.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import fsum, isfinite


def _signed_area(points: tuple[tuple[float, float], ...]) -> float:
    # Translate the calculation to reduce cancellation for offset outlines.
    ox, oy = points[0]
    return fsum(
        (x - ox) * (ny - oy) - (nx - ox) * (y - oy)
        for (x, y), (nx, ny) in zip(points, points[1:] + points[:1])
    ) / 2.0


@dataclass(frozen=True)
class CanonicalOutline:
    """An immutable, unclosed vertex ring with positive signed area.

    Only duplicate removal, winding reversal and cyclic rotation change the
    ring. This contract does not validate self-intersections or model holes.
    """

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        vertices = []
        for x, y in self.points:
            point = (float(x), float(y))
            if not all(isfinite(value) for value in point):
                raise ValueError("Outline coordinates must be finite")
            if not vertices or point != vertices[-1]:
                vertices.append(point)
        if len(vertices) > 1 and vertices[-1] == vertices[0]:
            vertices.pop()
        if len(set(vertices)) < 3:
            raise ValueError("An outline requires at least three distinct vertices")

        # Canonicalize the calculation order as well as the stored ring.
        start = min(range(len(vertices)), key=vertices.__getitem__)
        vertices = vertices[start:] + vertices[:start]
        area = _signed_area(tuple(vertices))
        if not isfinite(area) or area == 0:
            raise ValueError("An outline requires a finite, nonzero area")
        if area < 0:
            vertices.reverse()

        # Comparing complete rotations breaks ties between repeated minima.
        smallest = min(vertices)
        ring = min(
            tuple(vertices[i:] + vertices[:i])
            for i, point in enumerate(vertices)
            if point == smallest
        )
        object.__setattr__(self, "points", ring)

    @classmethod
    def from_points(cls, points: Iterable[tuple[float, float]]) -> "CanonicalOutline":
        """Copy and normalize an open or explicitly closed input ring."""
        return cls(tuple(points))

    @property
    def signed_area(self) -> float:
        return _signed_area(self.points)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs, ys = zip(*self.points)
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def width(self) -> float:
        min_x, _, max_x, _ = self.bounds
        return max_x - min_x

    @property
    def height(self) -> float:
        _, min_y, _, max_y = self.bounds
        return max_y - min_y
