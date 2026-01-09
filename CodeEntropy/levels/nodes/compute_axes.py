from CodeEntropy.levels.coordinate_system import CoordinateSystem


class ComputeAxesNode:
    def __init__(self):
        self._coord = CoordinateSystem()

    def run(self, shared_data):
        beads = shared_data["beads"]

        axes = {}
        avg_pos = {}

        for key, bead_list in beads.items():
            axes[key] = []
            avg_pos[key] = []

            for bead in bead_list:
                avg_pos[key].append(self._coord.get_avg_pos(bead))
                axes[key].append(self._coord.get_axes(bead))

        shared_data["axes"] = axes
        shared_data["avg_pos"] = avg_pos
