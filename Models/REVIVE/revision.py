


@dataclass
class Segment:
    tokens: torch.Tensor
    frame_ids: torch.Tensor
    basis: torch.Tensor


def make_segment(tokens: torch.Tensor, frame_ids: torch.Tensor, rank: int) -> Segment:
    basis = fit_basis(tokens, rank).basis
    return Segment(tokens=tokens, frame_ids=frame_ids, basis=basis)


@torch.no_grad()
def hypothesis_objective_components(
    segments: list[Segment],
    event_penalty: float,
    witness_penalty: float,
) -> dict[str, float]:
    residual_square_sum = 0.0
    residual_count = 0
    for segment in segments:
        residual = residual_scores(segment.tokens, segment.basis, relative=True)
        if residual.numel():
            residual_square_sum += float(residual.square().sum().item())
            residual_count += int(residual.numel())
    data_fit = residual_square_sum / max(residual_count, 1)
    event_cost = float(event_penalty) * len(segments)
    witness_cost = 0.0
    for left, right in zip(segments[:-1], segments[1:]):
        if left.tokens.numel() == 0 or right.tokens.numel() == 0:
            continue
        boundary_frame = int(right.frame_ids.min().item())
        left_near = left.frame_ids >= boundary_frame - 1
        right_near = right.frame_ids <= boundary_frame + 1
        witness_tokens = torch.cat([left.tokens[left_near], right.tokens[right_near]], dim=0)
        if witness_tokens.numel() == 0:
            continue
        left_residual = residual_scores(witness_tokens, left.basis, relative=True)
        right_residual = residual_scores(witness_tokens, right.basis, relative=True)
        ambiguity = torch.exp(-5.0 * (left_residual - right_residual).abs()).mean()
        witness_cost += float(witness_penalty) * float(ambiguity.item())
    return {
        "data_fit": float(data_fit),
        "event_cost": float(event_cost),
        "witness_cost": float(witness_cost),
        "total": float(data_fit + event_cost + witness_cost),
        "num_segments": float(len(segments)),
        "num_tokens": float(residual_count),
    }


@torch.no_grad()
def hypothesis_objective(segments: list[Segment], event_penalty: float, witness_penalty: float) -> float:
    return hypothesis_objective_components(segments, event_penalty, witness_penalty)["total"]


class RevisionEngine:
    def __init__(
        self,
        threshold: float,
        event_penalty: float,
        witness_penalty: float,
        local_rank: int,
        min_event_frames: int,
        old_candidates: int,
        reopen_similarity: float,
        merge_similarity: float = 0.90,
        enable_split: bool = True,
        enable_merge: bool = True,
        enable_reopen: bool = True,
        enable_move: bool = True,
        freeze_committed_events: bool = False,
        permute_delta_j_seed: Optional[int] = None,
    ) -> None:
        self.threshold = float(threshold)
        self.event_penalty = float(event_penalty)
        self.witness_penalty = float(witness_penalty)
        self.local_rank = int(local_rank)
        self.min_event_frames = int(min_event_frames)
        self.old_candidates = int(old_candidates)
        self.reopen_similarity = float(reopen_similarity)
        self.merge_similarity = float(merge_similarity)
        self.enable_split = enable_split
        self.enable_merge = enable_merge
        self.enable_reopen = enable_reopen
        self.enable_move = enable_move
        self.freeze_committed_events = bool(freeze_committed_events)
        self.permute_delta_j_seed = (
            None if permute_delta_j_seed is None else int(permute_delta_j_seed)
        )
        self.decision_index = 0

    @torch.no_grad()
    def choose(
        self,
        events: list[EventVersion],
        new_tokens: torch.Tensor,
        new_frame_ids: torch.Tensor,
        oracle_reopen_target_event_id: Optional[int] = None,
    ) -> RevisionDecision:
        self.decision_index += 1
        new_segment = make_segment(new_tokens, new_frame_ids, self.local_rank)
        event_segments = []
        for event in events:
            event_tokens, event_ids, _ = event.all_tokens()
            # Structural hypotheses must be compared at the same local rank.
            # Stored event bases use dynamically allocated ranks and can become
            # full-dimensional when only one event exists, which makes both
            # residuals and subspace similarity artificially perfect.
            event_segments.append(make_segment(event_tokens, event_ids, self.local_rank))
        current_segments = event_segments + [new_segment]
        current_components = hypothesis_objective_components(
            current_segments,
            self.event_penalty,
            self.witness_penalty,
        )
        current_score = current_components["total"]
        candidates: list[tuple[RevisionOperation, float, dict]] = []
        candidate_diagnostics: list[dict] = []

        def add_candidate(
            operation: RevisionOperation,
            segments: list[Segment],
            metadata: dict,
        ) -> None:
            if self.freeze_committed_events and metadata.get("target_event_ids"):
                candidate_diagnostics.append({
                    "operation": operation.value,
                    "eligible": False,
                    "reason": "committed_event_is_frozen",
                    **metadata,
                })
                return
            components = hypothesis_objective_components(
                segments,
                self.event_penalty,
                self.witness_penalty,
            )
            score = components["total"]
            diagnostic = {
                "operation": operation.value,
                "eligible": True,
                "score": score,
                "delta_J": current_score - score,
                "components": components,
                **metadata,
            }
            candidate_diagnostics.append(diagnostic)
            candidates.append((operation, score, metadata))

        if self.enable_merge and events:
            previous_tokens, previous_ids, _ = events[-1].all_tokens()
            similarity = subspace_similarity(event_segments[-1].basis, new_segment.basis)
            if similarity >= self.merge_similarity:
                merged = make_segment(
                    torch.cat([previous_tokens, new_tokens], dim=0),
                    torch.cat([previous_ids, new_frame_ids], dim=0),
                    self.local_rank,
                )
                add_candidate(
                    RevisionOperation.MERGE,
                    event_segments[:-1] + [merged],
                    {
                        "target_event_ids": [events[-1].event_id],
                        "similarity": similarity,
                        "similarity_threshold": self.merge_similarity,
                    },
                )
            else:
                candidate_diagnostics.append({
                    "operation": RevisionOperation.MERGE.value,
                    "eligible": False,
                    "reason": "subspace_similarity_below_threshold",
                    "similarity": similarity,
                    "similarity_threshold": self.merge_similarity,
                })

        new_unique_frames = torch.unique(new_frame_ids, sorted=True)
        unique_frames = new_unique_frames
        split_tokens = new_tokens
        split_ids = new_frame_ids
        split_source = "new"
        split_prefix = event_segments
        split_suffix = []
        split_target_ids = []
        if events:
            previous_tokens, previous_ids, _ = events[-1].all_tokens()
            previous_frames = torch.unique(previous_ids, sorted=True)
            if previous_frames.numel() >= 2 * self.min_event_frames:
                split_tokens = previous_tokens
                split_ids = previous_ids
                unique_frames = previous_frames
                split_source = "previous"
                split_prefix = event_segments[:-1]
                split_suffix = [new_segment]
                split_target_ids = [events[-1].event_id]
        if self.enable_split and unique_frames.numel() >= 2 * self.min_event_frames:
            for split_index in range(self.min_event_frames, unique_frames.numel() - self.min_event_frames + 1):
                split_frame = int(unique_frames[split_index].item())
                left_mask = split_ids < split_frame
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                split_segments = split_prefix + [
                    make_segment(split_tokens[left_mask], split_ids[left_mask], self.local_rank),
                    make_segment(split_tokens[right_mask], split_ids[right_mask], self.local_rank),
                ] + split_suffix
                add_candidate(
                    RevisionOperation.SPLIT,
                    split_segments,
                    {
                        "split_frame": split_frame,
                        "split_source": split_source,
                        "target_event_ids": split_target_ids,
                    },
                )

        if self.enable_move and events and new_unique_frames.numel() >= self.min_event_frames + 1:
            previous_tokens, previous_ids, _ = events[-1].all_tokens()
            combined_tokens = torch.cat([previous_tokens, new_tokens], dim=0)
            combined_ids = torch.cat([previous_ids, new_frame_ids], dim=0)
            boundary = int(new_frame_ids.min().item())
            candidate_frames = torch.unique(combined_ids, sorted=True)
            for split_frame_tensor in candidate_frames:
                split_frame = int(split_frame_tensor.item())
                if split_frame == boundary or abs(split_frame - boundary) > 2:
                    continue
                left_mask = combined_ids < split_frame
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                add_candidate(
                    RevisionOperation.MOVE,
                    event_segments[:-1] + [
                        make_segment(combined_tokens[left_mask], combined_ids[left_mask], self.local_rank),
                        make_segment(combined_tokens[right_mask], combined_ids[right_mask], self.local_rank),
                    ],
                    {
                        "split_frame": split_frame,
                        "target_event_ids": [events[-1].event_id],
                    },
                )

        if self.enable_reopen and len(events) > 1:
            similarities = []
            for event, event_segment in zip(events[:-1], event_segments[:-1]):
                similarities.append((subspace_similarity(event_segment.basis, new_segment.basis), event))
            if oracle_reopen_target_event_id is not None:
                selected_events = [
                    (similarity, event, True)
                    for similarity, event in similarities
                    if event.event_id == int(oracle_reopen_target_event_id)
                ]
            else:
                selected_events = [
                    (similarity, event, False)
                    for similarity, event in sorted(
                        similarities,
                        key=lambda item: item[0],
                        reverse=True,
                    )[:self.old_candidates]
                ]
            for similarity, event, oracle_targeted in selected_events:
                if similarity < self.reopen_similarity and not oracle_targeted:
                    continue
                old_tokens, old_ids, _ = event.all_tokens()
                reopened = make_segment(
                    torch.cat([old_tokens, new_tokens], dim=0),
                    torch.cat([old_ids, new_frame_ids], dim=0),
                    self.local_rank,
                )
                reopened_segments = list(event_segments)
                target_index = next(
                    index for index, other_event in enumerate(events)
                    if other_event.event_id == event.event_id
                )
                reopened_segments[target_index] = reopened
                add_candidate(
                    RevisionOperation.REOPEN,
                    reopened_segments,
                    {
                        "target_event_ids": [event.event_id],
                        "similarity": similarity,
                        "similarity_threshold": self.reopen_similarity,
                        "oracle_targeted": oracle_targeted,
                    },
                )

        compact_diagnostics = []
        for candidate_operation in RevisionOperation:
            rows = [
                row for row in candidate_diagnostics
                if row.get("operation") == candidate_operation.value
            ]
            if not rows:
                continue
            eligible_rows = [row for row in rows if row.get("eligible")]
            if eligible_rows:
                best = dict(min(eligible_rows, key=lambda row: float(row["score"])))
                best["num_hypotheses"] = len(eligible_rows)
                compact_diagnostics.append(best)
            else:
                compact_diagnostics.append(dict(rows[0]))

        if not candidates:
            return RevisionDecision(
                RevisionOperation.KEEP,
                current_score,
                current_score,
                0.0,
                False,
                metadata={
                    "current_components": current_components,
                    "candidate_scores": compact_diagnostics,
                    "selected_candidate_operation": RevisionOperation.KEEP.value,
                },
            )
        best_by_operation: dict[RevisionOperation, tuple[RevisionOperation, float, dict]] = {}
        for candidate in candidates:
            operation = candidate[0]
            previous = best_by_operation.get(operation)
            if previous is None or candidate[1] < previous[1]:
                best_by_operation[operation] = candidate
        selection_candidates = list(best_by_operation.values())
        permutation_metadata = None
        if self.permute_delta_j_seed is not None and len(selection_candidates) > 1:
            original_scores = [candidate[1] for candidate in selection_candidates]
            permuted_scores = list(original_scores)
            random.Random(
                self.permute_delta_j_seed + self.decision_index
            ).shuffle(permuted_scores)
            selection_candidates = [
                (operation, permuted_score, {
                    **metadata,
                    "true_candidate_score": true_score,
                })
                for (operation, true_score, metadata), permuted_score
                in zip(selection_candidates, permuted_scores)
            ]
            permutation_metadata = {
                candidate[0].value: {
                    "true_score": original_score,
                    "permuted_selection_score": permuted_score,
                }
                for candidate, original_score, permuted_score in zip(
                    selection_candidates,
                    original_scores,
                    permuted_scores,
                )
            }
            for diagnostic in compact_diagnostics:
                operation_name = str(diagnostic.get("operation", ""))
                if operation_name in permutation_metadata:
                    diagnostic["permuted_selection_score"] = permutation_metadata[
                        operation_name
                    ]["permuted_selection_score"]

        operation, selected_score, metadata = min(
            selection_candidates,
            key=lambda item: item[1],
        )
        delta_j = current_score - selected_score
        accepted = delta_j > self.threshold
        decision_metadata = {
            **metadata,
            "current_components": current_components,
            "candidate_scores": compact_diagnostics,
            "selected_candidate_operation": operation.value,
            "revision_threshold": self.threshold,
            "permuted_delta_j_seed": self.permute_delta_j_seed,
            "permuted_candidate_scores": permutation_metadata,
        }
        return RevisionDecision(
            operation=operation if accepted else RevisionOperation.KEEP,
            current_score=current_score,
            selected_score=selected_score,
            delta_j=delta_j,
            accepted=accepted,
            target_event_ids=list(metadata.get("target_event_ids", [])),
            split_frame=metadata.get("split_frame"),
            metadata=decision_metadata,
        )

