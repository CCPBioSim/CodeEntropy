import logging

from CodeEntropy.levels.coordinate_system import CoordinateSystem

logger = logging.getLogger(__name__)


class ComputeAxesNode:
    def __init__(self):
        self._coord = CoordinateSystem()

    def run(self, shared_data):
        beads = shared_data["beads"]

        trans_axes = {}
        rot_axes = {}
        avg_pos = {}

        for key, bead_list in beads.items():
            trans_axes[key] = []
            rot_axes[key] = []
            avg_pos[key] = []

            for bead in bead_list:
                t_ax, r_ax = self._coord.get_axes(
                    bead.data_container, bead.level, bead.index
                )

                trans_axes[key].append(t_ax)
                rot_axes[key].append(r_ax)
                avg_pos[key].append(
                    self._coord.get_avg_pos(bead.atoms, bead.atoms.center_of_mass())
                )

        shared_data["trans_axes"] = trans_axes
        shared_data["rot_axes"] = rot_axes
        shared_data["avg_positions"] = avg_pos
