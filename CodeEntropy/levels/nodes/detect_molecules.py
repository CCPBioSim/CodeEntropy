import logging

logger = logging.getLogger(__name__)


class DetectMoleculesNode:
    """
    Detect molecules in reduced_universe (so indices match all downstream work).
    """

    def run(self, shared_data):
        u = shared_data["reduced_universe"]
        fragments = u.atoms.fragments
        n_mol = len(fragments)

        logger.info(
            f"[DetectMoleculesNode] {n_mol} molecules detected (reduced_universe)"
        )

        shared_data["fragments"] = fragments
        shared_data["number_molecules"] = n_mol

        return {"number_molecules": n_mol}
