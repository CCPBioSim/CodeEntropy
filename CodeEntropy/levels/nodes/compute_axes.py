from CodeEntropy.levels.coordinate_system import CoordinateSystem


class ComputeAxesNode:
    def __init__(self):
        self._coord = CoordinateSystem()

    def run(self, shared_data, build_beads):
        axes = {}
        avg_pos = {}

        for key, bead_list in build_beads["beads"].items():
            axes[key] = []
            avg_pos[key] = []

            for bead in bead_list:
                avg_pos[key].append(self._coord.get_avg_pos(bead))
                axes[key].append(self._coord.get_axes(bead))

        return {"axes": axes, "avg_pos": avg_pos}
