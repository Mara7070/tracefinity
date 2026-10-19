"""Recovered Chappel dimensions in millimetres, without CAD dependencies."""

from dataclasses import dataclass


def _validate_cells(cells: int) -> None:
    if type(cells) is not int or cells < 1:
        raise ValueError("cells must be a Python int >= 1")


@dataclass(frozen=True)
class ChappelDimensions:
    """Footprint and height vocabulary for the 25, 40, and 75 series.

    Clearances apply once per footprint axis. The preferred interface names
    the later v2/40 interface; it does not describe mating-foot solids.
    """

    pitch: float = 55.0
    bin_footprint_clearance: float = 0.2
    organizer_footprint_clearance: float = 0.4
    organizer_height: float = 25.0
    short_height: float = 39.15
    full_height: float = 75.0
    preferred_interface_variant: str = "short40"

    def bin_footprint(self, cells: int) -> float:
        """Return the ordinary 40/75 footprint along one axis."""
        _validate_cells(cells)
        return self.pitch * cells - self.bin_footprint_clearance

    def organizer_footprint(self, cells: int) -> float:
        """Return the organizer/25-series footprint along one axis."""
        _validate_cells(cells)
        return self.pitch * cells - self.organizer_footprint_clearance

    def bin_footprint_xy(self, cells_x: int, cells_y: int) -> tuple[float, float]:
        """Return the ordinary footprint in (x, y) argument order."""
        return self.bin_footprint(cells_x), self.bin_footprint(cells_y)

    def organizer_footprint_xy(self, cells_x: int, cells_y: int) -> tuple[float, float]:
        """Return the organizer footprint in (x, y) argument order."""
        return self.organizer_footprint(cells_x), self.organizer_footprint(cells_y)


CHAPPEL = ChappelDimensions()
