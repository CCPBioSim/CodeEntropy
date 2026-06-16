"""Parent-side reducers for frame/chunk map-reduce outputs."""

from __future__ import annotations

from typing import Any

from CodeEntropy.levels.execution.tasks import CovarianceChunkPartial


def stable_keys(mapping: dict[Any, Any]) -> list[Any]:
    """Return mapping keys in deterministic order.

    Args:
        mapping: Mapping whose keys should be ordered independently of process hash
            randomisation.

    Returns:
        A list of keys sorted by key type name and representation.
    """
    return sorted(mapping.keys(), key=lambda key: (type(key).__name__, repr(key)))


def merge_means(old_mean: Any, old_n: int, new_mean: Any, new_n: int) -> Any:
    """Merge two running means using their sample counts.

    Args:
        old_mean: Existing mean value, or ``None`` if no samples have been seen.
        old_n: Number of samples represented by ``old_mean``.
        new_mean: New mean value to merge.
        new_n: Number of samples represented by ``new_mean``.

    Returns:
        The merged mean. If ``new_n`` is zero or negative, ``old_mean`` is returned.
    """
    if new_n <= 0:
        return old_mean
    if old_mean is None or old_n <= 0:
        return new_mean.copy() if hasattr(new_mean, "copy") else new_mean
    total_n = old_n + new_n
    return old_mean + (new_mean - old_mean) * (float(new_n) / float(total_n))


def incremental_mean(old: Any, new: Any, n: int) -> Any:
    """Update a running mean with one new sample.

    Args:
        old: Existing running mean, or ``None`` for the first sample.
        new: New sample to incorporate.
        n: One-based sample count after adding ``new``.

    Returns:
        The updated running mean.
    """
    if old is None:
        return new.copy() if hasattr(new, "copy") else new
    return old + (new - old) / float(n)


class NeighborReducer:
    """Initialise, merge, and finalise neighbour-count reductions."""

    @staticmethod
    def initialise(shared_data: dict[str, Any]) -> None:
        """Initialise parent-side neighbour accumulators.

        Args:
            shared_data: Shared workflow data containing ``groups``. The method writes
                ``neighbor_totals`` and ``neighbor_samples``.
        """
        shared_data["neighbor_totals"] = {
            group_id: 0 for group_id in shared_data["groups"].keys()
        }
        shared_data["neighbor_samples"] = {
            group_id: 0 for group_id in shared_data["groups"].keys()
        }

    @staticmethod
    def reduce_frame_output(
        shared_data: dict[str, Any],
        frame_neighbors: dict[int, tuple[int, int]] | None,
    ) -> None:
        """Merge one frame's neighbour-count payload.

        Args:
            shared_data: Shared workflow data containing neighbour total/sample
                accumulators.
            frame_neighbors: Optional mapping of group id to ``(count, sample_count)``.
        """
        if frame_neighbors is None:
            return

        totals = shared_data["neighbor_totals"]
        samples = shared_data["neighbor_samples"]
        for group_id in stable_keys(frame_neighbors):
            count, sample_count = frame_neighbors[group_id]
            totals[group_id] = totals.get(group_id, 0) + int(count)
            samples[group_id] = samples.get(group_id, 0) + int(sample_count)

    @staticmethod
    def merge_chunk_partial(
        shared_data: dict[str, Any],
        neighbor_totals: dict[int, int],
        neighbor_samples: dict[int, int],
    ) -> None:
        """Merge chunk-level neighbour totals and samples.

        Args:
            shared_data: Shared workflow data containing neighbour accumulators.
            neighbor_totals: Mapping of group id to additive neighbour totals.
            neighbor_samples: Mapping of group id to additive sample counts.
        """
        totals = shared_data.get("neighbor_totals")
        samples = shared_data.get("neighbor_samples")
        if totals is None or samples is None:
            return

        for group_id in stable_keys(neighbor_totals):
            count = neighbor_totals[group_id]
            totals[group_id] = totals.get(group_id, 0) + int(count)
        for group_id in stable_keys(neighbor_samples):
            sample_count = neighbor_samples[group_id]
            samples[group_id] = samples.get(group_id, 0) + int(sample_count)

    @staticmethod
    def finalise(shared_data: dict[str, Any]) -> None:
        """Compute average neighbour counts from reduced totals.

        Args:
            shared_data: Shared workflow data containing ``groups``,
                ``neighbor_totals``, and ``neighbor_samples``. The method writes
                ``neighbors``.
        """
        neighbors = {}
        for group_id in stable_keys(shared_data["groups"]):
            sample_count = shared_data["neighbor_samples"].get(group_id, 0)
            if sample_count <= 0:
                neighbors[group_id] = 0.0
            else:
                neighbors[group_id] = (
                    shared_data["neighbor_totals"].get(group_id, 0) / sample_count
                )
        shared_data["neighbors"] = neighbors


class CovarianceReducer:
    """Merge frame and chunk covariance outputs into canonical accumulators."""

    def reduce_frame_output(
        self,
        shared_data: dict[str, Any],
        frame_out: dict[str, Any],
    ) -> None:
        """Reduce one frame covariance payload into parent accumulators.

        Args:
            shared_data: Shared workflow data containing covariance accumulators.
            frame_out: Frame covariance payload with force, torque, and optional
                force-torque sections.
        """
        self._reduce_force_and_torque(shared_data, frame_out)
        self._reduce_forcetorque(shared_data, frame_out)

    def merge_chunk_partial(
        self,
        shared_data: dict[str, Any],
        partial: CovarianceChunkPartial,
    ) -> None:
        """Merge a worker covariance partial into parent accumulators.

        Args:
            shared_data: Shared workflow data containing covariance accumulators.
            partial: Compact covariance partial returned by a worker frame chunk.
        """
        self._merge_force_and_torque_partial(shared_data, partial)
        self._merge_forcetorque_partial(shared_data, partial)

    def reduce_frame_map_output(
        self,
        shared_data: dict[str, Any],
        frame_out: dict[str, Any],
    ) -> None:
        """Reduce a complete serial MAP output.

        Args:
            shared_data: Shared workflow data containing covariance and neighbour
                accumulators.
            frame_out: MAP output containing optional ``covariance`` and ``neighbors``
                entries.
        """
        covariance = frame_out.get("covariance")
        if covariance is not None:
            self.reduce_frame_output(shared_data, covariance)

        neighbors = frame_out.get("neighbors")
        if neighbors is not None:
            NeighborReducer.reduce_frame_output(shared_data, neighbors)

    def _merge_force_and_torque_partial(
        self,
        shared_data: dict[str, Any],
        partial: CovarianceChunkPartial,
    ) -> None:
        """Merge chunk force and torque means into parent accumulators.

        Args:
            shared_data: Shared workflow data containing force/torque accumulators,
                frame counts, and ``group_id_to_index``.
            partial: Worker covariance partial with force, torque, and count mappings.
        """
        f_cov = shared_data["force_covariances"]
        t_cov = shared_data["torque_covariances"]
        counts = shared_data["frame_counts"]
        gid2i = shared_data["group_id_to_index"]

        for key in stable_keys(partial.frame_counts["ua"]):
            partial_n = partial.frame_counts["ua"][key]
            old_n = int(counts["ua"].get(key, 0))
            if key in partial.force["ua"]:
                f_cov["ua"][key] = merge_means(
                    f_cov["ua"].get(key), old_n, partial.force["ua"][key], partial_n
                )
            if key in partial.torque["ua"]:
                t_cov["ua"][key] = merge_means(
                    t_cov["ua"].get(key), old_n, partial.torque["ua"][key], partial_n
                )
            counts["ua"][key] = old_n + partial_n

        for gid in stable_keys(partial.frame_counts["res"]):
            partial_n = partial.frame_counts["res"][gid]
            gi = gid2i[gid]
            old_n = int(counts["res"][gi])
            if gid in partial.force["res"]:
                f_cov["res"][gi] = merge_means(
                    f_cov["res"][gi], old_n, partial.force["res"][gid], partial_n
                )
            if gid in partial.torque["res"]:
                t_cov["res"][gi] = merge_means(
                    t_cov["res"][gi], old_n, partial.torque["res"][gid], partial_n
                )
            counts["res"][gi] = old_n + partial_n

        for gid in stable_keys(partial.frame_counts["poly"]):
            partial_n = partial.frame_counts["poly"][gid]
            gi = gid2i[gid]
            old_n = int(counts["poly"][gi])
            if gid in partial.force["poly"]:
                f_cov["poly"][gi] = merge_means(
                    f_cov["poly"][gi], old_n, partial.force["poly"][gid], partial_n
                )
            if gid in partial.torque["poly"]:
                t_cov["poly"][gi] = merge_means(
                    t_cov["poly"][gi], old_n, partial.torque["poly"][gid], partial_n
                )
            counts["poly"][gi] = old_n + partial_n

    @staticmethod
    def _merge_forcetorque_partial(
        shared_data: dict[str, Any],
        partial: CovarianceChunkPartial,
    ) -> None:
        """Merge chunk force-torque block means into parent accumulators.

        Args:
            shared_data: Shared workflow data containing force-torque accumulators,
                force-torque counts, and ``group_id_to_index``.
            partial: Worker covariance partial with force-torque matrices and counts.
        """
        ft_cov = shared_data["forcetorque_covariances"]
        ft_counts = shared_data["forcetorque_counts"]
        gid2i = shared_data["group_id_to_index"]

        for gid in stable_keys(partial.forcetorque_counts["res"]):
            partial_n = partial.forcetorque_counts["res"][gid]
            gi = gid2i[gid]
            old_n = int(ft_counts["res"][gi])
            ft_cov["res"][gi] = merge_means(
                ft_cov["res"][gi], old_n, partial.forcetorque["res"][gid], partial_n
            )
            ft_counts["res"][gi] = old_n + partial_n

        for gid in stable_keys(partial.forcetorque_counts["poly"]):
            partial_n = partial.forcetorque_counts["poly"][gid]
            gi = gid2i[gid]
            old_n = int(ft_counts["poly"][gi])
            ft_cov["poly"][gi] = merge_means(
                ft_cov["poly"][gi], old_n, partial.forcetorque["poly"][gid], partial_n
            )
            ft_counts["poly"][gi] = old_n + partial_n

    def _reduce_force_and_torque(
        self,
        shared_data: dict[str, Any],
        frame_out: dict[str, Any],
    ) -> None:
        """Reduce frame force and torque matrices into running means.

        Args:
            shared_data: Shared workflow data containing force/torque accumulators,
                frame counts, and ``group_id_to_index``.
            frame_out: Frame covariance payload with ``force`` and ``torque`` sections.
        """
        f_cov = shared_data["force_covariances"]
        t_cov = shared_data["torque_covariances"]
        counts = shared_data["frame_counts"]
        gid2i = shared_data["group_id_to_index"]

        f_frame = frame_out["force"]
        t_frame = frame_out["torque"]

        for key in stable_keys(f_frame["ua"]):
            F = f_frame["ua"][key]
            counts["ua"][key] = counts["ua"].get(key, 0) + 1
            n = counts["ua"][key]
            f_cov["ua"][key] = incremental_mean(f_cov["ua"].get(key), F, n)

        for key in stable_keys(t_frame["ua"]):
            T = t_frame["ua"][key]
            if key not in counts["ua"]:
                counts["ua"][key] = counts["ua"].get(key, 0) + 1
            n = counts["ua"][key]
            t_cov["ua"][key] = incremental_mean(t_cov["ua"].get(key), T, n)

        for gid in stable_keys(f_frame["res"]):
            F = f_frame["res"][gid]
            gi = gid2i[gid]
            counts["res"][gi] += 1
            n = counts["res"][gi]
            f_cov["res"][gi] = incremental_mean(f_cov["res"][gi], F, n)

        for gid in stable_keys(t_frame["res"]):
            T = t_frame["res"][gid]
            gi = gid2i[gid]
            if counts["res"][gi] == 0:
                counts["res"][gi] += 1
            n = counts["res"][gi]
            t_cov["res"][gi] = incremental_mean(t_cov["res"][gi], T, n)

        for gid in stable_keys(f_frame["poly"]):
            F = f_frame["poly"][gid]
            gi = gid2i[gid]
            counts["poly"][gi] += 1
            n = counts["poly"][gi]
            f_cov["poly"][gi] = incremental_mean(f_cov["poly"][gi], F, n)

        for gid in stable_keys(t_frame["poly"]):
            T = t_frame["poly"][gid]
            gi = gid2i[gid]
            if counts["poly"][gi] == 0:
                counts["poly"][gi] += 1
            n = counts["poly"][gi]
            t_cov["poly"][gi] = incremental_mean(t_cov["poly"][gi], T, n)

    def _reduce_forcetorque(
        self,
        shared_data: dict[str, Any],
        frame_out: dict[str, Any],
    ) -> None:
        """Reduce frame force-torque matrices into running means.

        Args:
            shared_data: Shared workflow data containing force-torque accumulators,
                force-torque counts, and ``group_id_to_index``.
            frame_out: Frame covariance payload that may contain a ``forcetorque``
                section.
        """
        if "forcetorque" not in frame_out:
            return

        ft_cov = shared_data["forcetorque_covariances"]
        ft_counts = shared_data["forcetorque_counts"]
        gid2i = shared_data["group_id_to_index"]
        ft_frame = frame_out["forcetorque"]

        for gid in stable_keys(ft_frame.get("res", {})):
            M = ft_frame["res"][gid]
            gi = gid2i[gid]
            ft_counts["res"][gi] += 1
            n = ft_counts["res"][gi]
            ft_cov["res"][gi] = incremental_mean(ft_cov["res"][gi], M, n)

        for gid in stable_keys(ft_frame.get("poly", {})):
            M = ft_frame["poly"][gid]
            gi = gid2i[gid]
            ft_counts["poly"][gi] += 1
            n = ft_counts["poly"][gi]
            ft_cov["poly"][gi] = incremental_mean(ft_cov["poly"][gi], M, n)
