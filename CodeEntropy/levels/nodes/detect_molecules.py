import logging

from CodeEntropy.group_molecules.group_molecules import GroupMolecules

logger = logging.getLogger(__name__)


class DetectMoleculesNode:
    """
    Establish shared_data['reduced_universe'] and shared_data['groups'].

    Assumptions (matches what you've been running):
      - shared_data already contains 'universe'
      - reduced_universe is either already present or is the same as universe
      - grouping uses your existing GroupMolecules implementation
    """

    def __init__(self):
        self._group = GroupMolecules()

    def run(self, shared_data):
        u = shared_data.get("reduced_universe", None)
        if u is None:
            u = shared_data.get("universe", None)
            if u is None:
                raise KeyError("shared_data must contain 'universe'")
            shared_data["reduced_universe"] = u

        args = shared_data["args"]
        grouping = getattr(args, "grouping", "each")

        groups = self._group.grouping_molecules(u, grouping)
        shared_data["groups"] = groups
        shared_data["number_molecules"] = len(u.atoms.fragments)

        logger.info(
            f"[DetectMoleculesNode] {shared_data['number_molecules']} "
            "molecules detected (reduced_universe)"
        )
        return {"groups": groups, "number_molecules": shared_data["number_molecules"]}
