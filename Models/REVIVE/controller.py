from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Optional

import torch

from Models.Memory.base import MemoryOutput, StreamingMemoryController
from Models.REVIVE.boundary import RobustBoundaryScorer
from Models.REVIVE.budget import allocate_event_budgets, prune_event
from Models.REVIVE.revision import RevisionEngine
from Models.REVIVE.state import (
    EventState,
    EventVersion,
    FramePacket,
    RevisionDecision,
    RevisionOperation,
)
from Models.REVIVE.subspace import empty_basis, fit_basis, residual_scores, update_basis
from Models.REVIVE.witness import select_boundary_witnesses


class ReviveMemory(StreamingMemoryController):
    """Training-free, fixed-budget, versioned event memory.

    The controller sees projected visual tokens only. It owns no trainable parameter
    and never calls the language model while the stream is being updated.
    """

    def __init__(
        self,
        *args,
        rank_budget: int = 256,
        pel_frames: int = 8,
        pel_token_ratio: float = 0.25,
        boundary_threshold: float = 0.72,
        boundary_persist: int = 2,
        revision_threshold: float = 0.03,
        event_penalty: float = 0.02,
        witness_penalty: float = 0.05,
        witness_ratio: float = 0.15,
        min_event_tokens: int = 16,
        min_event_rank: int = 2,
        min_event_frames: int = 2,
        local_rank: int = 16,
        old_candidates: int = 1,
        reopen_similarity: float = 0.82,
        merge_similarity: float = 0.90,
        budget_temperature: float = 0.7,
        enable_split: bool = True,
        enable_merge: bool = True,
        enable_reopen: bool = True,
        enable_move: bool = True,
        enable_revision: bool = True,
        enable_pel: bool = True,
        enable_witnesses: bool = True,
        random_witness: bool = False,
        uniform_budget: bool = False,
        delete_witnesses_at_query: bool = False,
        misindex_witnesses: bool = False,
        freeze_committed_events: bool = False,
        permute_delta_j_seed: Optional[int] = None,
        use_oracle_event_target: bool = False,
        oracle_query_timestamp_seconds: Optional[float] = None,
        oracle_event_intervals_seconds: Optional[list[list[float]]] = None,
        active_basis_decay: float = 0.9,
        active_max_new_basis: int = 4,
        reject_revision_attempts: Optional[list[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rank_budget = int(rank_budget)
        self.pel_frames = max(2, int(pel_frames))
        self.enable_pel = bool(enable_pel)
        self.pel_token_budget = (
            max(1, int(round(self.token_budget * float(pel_token_ratio))))
            if self.enable_pel else 0
        )
        self.boundary_threshold = float(boundary_threshold)
        self.boundary_persist = max(1, int(boundary_persist))
        self.witness_ratio = float(witness_ratio)
        self.min_event_tokens = int(min_event_tokens)
        self.min_event_rank = int(min_event_rank)
        self.local_rank = int(local_rank)
        self.budget_temperature = float(budget_temperature)
        self.enable_revision = bool(enable_revision)
        self.enable_witnesses = bool(enable_witnesses)
        self.random_witness = bool(random_witness)
        self.uniform_budget = bool(uniform_budget)
        self.delete_witnesses_at_query = bool(delete_witnesses_at_query)
        self.misindex_witnesses = bool(misindex_witnesses)
        self.use_oracle_event_target = bool(use_oracle_event_target)
        self.oracle_query_timestamp_seconds = (
            None
            if oracle_query_timestamp_seconds is None
            else float(oracle_query_timestamp_seconds)
        )
        self.oracle_event_intervals_seconds = [
            (float(interval[0]), float(interval[1]))
            for interval in (oracle_event_intervals_seconds or [])
            if len(interval) == 2
        ]
        self.active_basis_decay = float(active_basis_decay)
        self.active_max_new_basis = int(active_max_new_basis)
        self.reject_revision_attempts = {
            int(attempt) for attempt in (reject_revision_attempts or [])
        }
        if any(attempt <= 0 for attempt in self.reject_revision_attempts):
            raise ValueError("reject_revision_attempts must contain positive 1-based indices")

        self.events: list[EventVersion] = []
        self.pel: deque[FramePacket] = deque()
        self.boundary_scorer = RobustBoundaryScorer(history_size=max(8, self.pel_frames * 3))
        self.active_basis_state = empty_basis(self.dim, self.device, self.dtype)
        self.high_boundary_run = 0
        self.next_event_id = 1
        self.revision_attempt_count = 0
        self.frame_timestamps: dict[int, float] = {}
        self.oracle_future_frames_used = 0
        self._oracle_sanitized = False
        self.last_decision = RevisionDecision(RevisionOperation.KEEP, 0.0, 0.0, 0.0, False)
        self.revision_engine = RevisionEngine(
            threshold=revision_threshold,
            event_penalty=event_penalty,
            witness_penalty=witness_penalty,
            local_rank=local_rank,
            min_event_frames=min_event_frames,
            old_candidates=old_candidates,
            reopen_similarity=reopen_similarity,
            merge_similarity=merge_similarity,
            enable_split=enable_split,
            enable_merge=enable_merge,
            enable_reopen=enable_reopen,
            enable_move=enable_move,
            freeze_committed_events=freeze_committed_events,
            permute_delta_j_seed=permute_delta_j_seed,
        )

    @staticmethod
    def _intervals(frame_ids: torch.Tensor) -> list[tuple[int, int]]:
        if frame_ids.numel() == 0:
            return []
        frames = torch.unique(frame_ids.long(), sorted=True).cpu().tolist()
        intervals = []
        start = previous = int(frames[0])
        for frame in frames[1:]:
            frame = int(frame)
            if frame != previous + 1:
                intervals.append((start, previous))
                start = frame
            previous = frame
        intervals.append((start, previous))
        return intervals

    def _new_event(
        self,
        tokens: torch.Tensor,
        frame_ids: torch.Tensor,
        event_id: Optional[int] = None,
        version: int = 1,
        parents: Optional[list[str]] = None,
        operation: RevisionOperation = RevisionOperation.KEEP,
        state: EventState = EventState.COMMITTED,
        revision_count: int = 0,
    ) -> EventVersion:
        if event_id is None:
            event_id = self.next_event_id
            self.next_event_id += 1
        basis_state = fit_basis(tokens, min(self.local_rank, self.rank_budget))
        residual = residual_scores(tokens, basis_state.basis, relative=True)
        empty_witness = torch.empty((0, self.dim), device=self.device, dtype=self.dtype)
        empty_ids = torch.empty((0,), device=self.device, dtype=torch.long)
        return EventVersion(
            event_id=event_id,
            version=version,
            intervals=self._intervals(frame_ids),
            basis=basis_state.basis,
            strengths=basis_state.strengths,
            core_tokens=tokens,
            core_frame_ids=frame_ids,
            witness_tokens=empty_witness,
            witness_frame_ids=empty_ids,
            state=state,
            confidence=max(0.0, 1.0 - float(residual.mean().item()) if residual.numel() else 0.5),
            residual_mean=float(residual.mean().item()) if residual.numel() else 0.0,
            revision_count=revision_count,
            parent_versions=parents or [],
            operation=operation,
        )

    @staticmethod
    def _pack_frames(frames: Iterable[FramePacket]) -> tuple[torch.Tensor, torch.Tensor]:
        frames = list(frames)
        if not frames:
            raise ValueError("Cannot pack an empty frame list")
        tokens = torch.cat([frame.tokens for frame in frames], dim=0)
        frame_ids = torch.cat([
            torch.full((frame.tokens.shape[0],), frame.frame_id, device=frame.tokens.device, dtype=torch.long)
            for frame in frames
        ])
        return tokens, frame_ids

    def _active_basis(self) -> torch.Tensor:
        return self.active_basis_state.basis

    def _reset_active_basis(self) -> None:
        if not self.pel:
            self.active_basis_state = empty_basis(self.dim, self.device, self.dtype)
            return
        tokens, _ = self._pack_frames(self.pel)
        self.active_basis_state = fit_basis(tokens, min(self.local_rank, self.rank_budget))

    @torch.no_grad()
    def _prune_pel(self) -> None:
        total = sum(frame.tokens.shape[0] for frame in self.pel)
        if total <= self.pel_token_budget or not self.pel:
            return
        basis = self._active_basis()
        per_frame = max(1, self.pel_token_budget // len(self.pel))
        packets = list(self.pel)
        for packet in packets:
            if packet.tokens.shape[0] <= per_frame:
                continue
            scores = residual_scores(packet.tokens, basis, relative=False)
            indices = torch.topk(scores, k=per_frame).indices
            packet.tokens = packet.tokens[indices]
        self.pel = deque(packets)
        while sum(frame.tokens.shape[0] for frame in self.pel) > self.pel_token_budget:
            largest = max(self.pel, key=lambda packet: packet.tokens.shape[0])
            largest.tokens = largest.tokens[:-1]

    @torch.no_grad()
    def process_frame(self, tokens: torch.Tensor, frame_idx: int, timestamp: Optional[float] = None) -> None:
        update_start = self._start_update_timer()
        tokens = self._prepare_tokens(tokens)
        active_basis = self._active_basis()
        novelty, boundary_q = self.boundary_scorer.score(tokens, active_basis)
        packet = FramePacket(
            tokens=tokens,
            frame_id=int(frame_idx),
            timestamp=float(frame_idx if timestamp is None else timestamp),
            boundary_q=boundary_q,
            novelty=novelty,
        )
        self.frame_timestamps[int(frame_idx)] = packet.timestamp
        self.pel.append(packet)
        if self.enable_pel:
            self._prune_pel()

        self.high_boundary_run = self.high_boundary_run + 1 if boundary_q >= self.boundary_threshold else 0
        operation = "buffer"
        decision = RevisionDecision(RevisionOperation.KEEP, 0.0, 0.0, 0.0, False)

        if not self.enable_pel:
            decision = self._consolidate_first(len(self.pel))
            operation = decision.operation.value if decision.accepted else "commit"
            self.high_boundary_run = 0
        elif self.high_boundary_run >= self.boundary_persist:
            commit_count = len(self.pel) - self.high_boundary_run
            if commit_count >= self.revision_engine.min_event_frames:
                decision = self._consolidate_first(commit_count)
                operation = decision.operation.value if decision.accepted else "commit"
                self.high_boundary_run = 0
        elif len(self.pel) > self.pel_frames:
            keep_recent = max(self.revision_engine.min_event_frames, self.pel_frames // 2)
            commit_count = len(self.pel) - keep_recent
            if commit_count > 0:
                decision = self._consolidate_first(commit_count)
                operation = decision.operation.value if decision.accepted else "commit"

        self.last_decision = decision
        if operation == "buffer":
            self.active_basis_state = update_basis(
                self.active_basis_state,
                tokens,
                max_rank=min(self.local_rank, self.rank_budget),
                decay=self.active_basis_decay,
                max_new_basis=self.active_max_new_basis,
            )
        else:
            self._reset_active_basis()
        self._enforce_budget()
        update_ms = self._finish_update_timer(update_start)
        visible_tokens = sum(event.visible_tokens for event in self.events) + sum(frame.tokens.shape[0] for frame in self.pel)
        eligible_operations = list(decision.metadata.get("eligible_operations", []))
        self.trace_records.append({
            "frame_id": int(frame_idx),
            "timestamp": packet.timestamp,
            "operation": operation,
            "revision_accepted": bool(decision.accepted),
            "revision_candidate_operation": decision.metadata.get(
                "selected_candidate_operation",
                decision.operation.value,
            ),
            "revision_target_event_ids": list(decision.target_event_ids),
            "revision_split_frame": decision.split_frame,
            "revision_current_components": decision.metadata.get("current_components", {}),
            "revision_candidate_scores": decision.metadata.get("candidate_scores", []),
            "revision_attempt": decision.metadata.get("revision_attempt"),
            "revision_eligible": bool(eligible_operations),
            "revision_eligible_operations": eligible_operations,
            "revision_forced_reject": bool(decision.metadata.get("forced_reject", False)),
            "delta_J": float(decision.delta_j),
            "J_current": float(decision.current_score),
            "J_selected": float(decision.selected_score),
            "boundary_q": float(boundary_q),
            "event_local_novelty": float(novelty),
            "num_events": len(self.events),
            "event_versions": [event.trace_dict() for event in self.events],
            "token_budget": self.token_budget,
            "rank_budget": self.rank_budget,
            "visible_tokens": int(min(visible_tokens, self.token_budget)),
            "rank_used": int(self.active_basis_state.rank + sum(event.rank for event in self.events)),
            "pel_tokens": int(sum(frame.tokens.shape[0] for frame in self.pel)),
            "pel_frame_ids": sorted({int(frame.frame_id) for frame in self.pel}),
            "witness_tokens": int(sum(event.witness_tokens.shape[0] for event in self.events)),
            "update_ms": update_ms,
        })

    @torch.no_grad()
    def _consolidate_first(self, frame_count: int) -> RevisionDecision:
        frames = [self.pel.popleft() for _ in range(min(frame_count, len(self.pel)))]
        tokens, frame_ids = self._pack_frames(frames)
        decision = self.revision_engine.choose(
            self.events,
            tokens,
            frame_ids,
            oracle_reopen_target_event_id=self._oracle_reopen_target_event_id(frame_ids),
        ) if self.enable_revision else RevisionDecision(
            RevisionOperation.KEEP, 0.0, 0.0, 0.0, False
        )
        eligible_operations = sorted({
            str(candidate.get("operation", "")).lower()
            for candidate in decision.metadata.get("candidate_scores", [])
            if candidate.get("eligible")
            and str(candidate.get("operation", "")).lower() in {"move", "reopen", "split", "merge"}
        })
        metadata = dict(decision.metadata)
        metadata["eligible_operations"] = eligible_operations
        revision_attempt = None
        if eligible_operations:
            self.revision_attempt_count += 1
            revision_attempt = self.revision_attempt_count
        metadata["revision_attempt"] = revision_attempt

        forced_reject = bool(
            decision.accepted
            and revision_attempt is not None
            and revision_attempt in self.reject_revision_attempts
        )
        metadata["forced_reject"] = forced_reject
        if forced_reject:
            metadata["forced_reject_original_operation"] = decision.operation.value
            decision = RevisionDecision(
                operation=decision.operation,
                current_score=decision.current_score,
                selected_score=decision.selected_score,
                delta_j=decision.delta_j,
                accepted=False,
                target_event_ids=list(decision.target_event_ids),
                split_frame=decision.split_frame,
                metadata=metadata,
            )
        else:
            decision.metadata = metadata
        self._apply_decision(decision, tokens, frame_ids)
        new_start = int(frame_ids.min().item())
        new_end = int(frame_ids.max().item())
        for event in self.events:
            touches_new_segment = any(
                not (interval_end < new_start or interval_start > new_end)
                for interval_start, interval_end in event.intervals
            )
            if not touches_new_segment:
                continue
            if event.start_frame >= new_start:
                event.start_boundary_q = float(frames[0].boundary_q)
            event.end_boundary_q = float(frames[-1].boundary_q)
        self._refresh_witnesses()
        self.boundary_scorer.reset([packet.novelty for packet in self.pel])
        return decision

    @torch.no_grad()
    def _apply_decision(self, decision: RevisionDecision, tokens: torch.Tensor, frame_ids: torch.Tensor) -> None:
        operation = decision.operation if decision.accepted else RevisionOperation.KEEP
        if operation == RevisionOperation.KEEP or not self.events:
            self.events.append(self._new_event(tokens, frame_ids))
            return

        if operation == RevisionOperation.MERGE:
            previous = self.events.pop()
            previous_tokens, previous_ids, _ = previous.all_tokens()
            merged_tokens = torch.cat([previous_tokens, tokens], dim=0)
            merged_ids = torch.cat([previous_ids, frame_ids], dim=0)
            self.events.append(self._new_event(
                merged_tokens,
                merged_ids,
                event_id=previous.event_id,
                version=previous.version + 1,
                parents=[previous.key],
                operation=RevisionOperation.MERGE,
                state=EventState.REVISED,
                revision_count=previous.revision_count + 1,
            ))
            return

        if operation == RevisionOperation.REOPEN and decision.target_event_ids:
            target_id = decision.target_event_ids[0]
            for index, event in enumerate(self.events):
                if event.event_id != target_id:
                    continue
                old_tokens, old_ids, _ = event.all_tokens()
                self.events[index] = self._new_event(
                    torch.cat([old_tokens, tokens], dim=0),
                    torch.cat([old_ids, frame_ids], dim=0),
                    event_id=event.event_id,
                    version=event.version + 1,
                    parents=[event.key],
                    operation=RevisionOperation.REOPEN,
                    state=EventState.REVISED,
                    revision_count=event.revision_count + 1,
                )
                return

        if operation == RevisionOperation.SPLIT and decision.split_frame is not None:
            if decision.metadata.get("split_source") == "previous" and self.events:
                previous = self.events.pop()
                previous_tokens, previous_ids, _ = previous.all_tokens()
                left_mask = previous_ids < decision.split_frame
                right_mask = ~left_mask
                if left_mask.any() and right_mask.any():
                    self.events.append(self._new_event(
                        previous_tokens[left_mask],
                        previous_ids[left_mask],
                        event_id=previous.event_id,
                        version=previous.version + 1,
                        parents=[previous.key],
                        operation=RevisionOperation.SPLIT,
                        state=EventState.REVISED,
                        revision_count=previous.revision_count + 1,
                    ))
                    self.events.append(self._new_event(
                        previous_tokens[right_mask],
                        previous_ids[right_mask],
                        parents=[previous.key],
                        operation=RevisionOperation.SPLIT,
                        state=EventState.REVISED,
                    ))
                    self.events.append(self._new_event(tokens, frame_ids))
                    return
                self.events.append(previous)
            left_mask = frame_ids < decision.split_frame
            right_mask = ~left_mask
            if left_mask.any() and right_mask.any():
                self.events.append(self._new_event(tokens[left_mask], frame_ids[left_mask], operation=RevisionOperation.SPLIT))
                self.events.append(self._new_event(tokens[right_mask], frame_ids[right_mask], operation=RevisionOperation.SPLIT))
                return

        if operation == RevisionOperation.MOVE and decision.split_frame is not None:
            previous = self.events.pop()
            previous_tokens, previous_ids, _ = previous.all_tokens()
            combined_tokens = torch.cat([previous_tokens, tokens], dim=0)
            combined_ids = torch.cat([previous_ids, frame_ids], dim=0)
            left_mask = combined_ids < decision.split_frame
            right_mask = ~left_mask
            if left_mask.any() and right_mask.any():
                self.events.append(self._new_event(
                    combined_tokens[left_mask],
                    combined_ids[left_mask],
                    event_id=previous.event_id,
                    version=previous.version + 1,
                    parents=[previous.key],
                    operation=RevisionOperation.MOVE,
                    state=EventState.REVISED,
                    revision_count=previous.revision_count + 1,
                ))
                self.events.append(self._new_event(
                    combined_tokens[right_mask], combined_ids[right_mask], operation=RevisionOperation.MOVE
                ))
                return
            self.events.append(previous)

        self.events.append(self._new_event(tokens, frame_ids))

    @torch.no_grad()
    def _refresh_witnesses(self) -> None:
        if not self.events:
            return
        if not self.enable_witnesses:
            for event in self.events:
                event.witness_tokens = torch.empty(
                    (0, self.dim), device=self.device, dtype=self.dtype
                )
                event.witness_frame_ids = torch.empty(
                    (0,), device=self.device, dtype=torch.long
                )
            return
        candidate_tokens = {}
        candidate_ids = {}
        selected_indices: dict[int, list[torch.Tensor]] = {event.event_id: [] for event in self.events}
        for event in self.events:
            tokens, frame_ids, _ = event.all_tokens()
            candidate_tokens[event.event_id] = tokens
            candidate_ids[event.event_id] = frame_ids
        per_boundary = max(2, int(math.ceil(self.token_budget * self.witness_ratio / max(len(self.events) - 1, 1))))
        for left, right in zip(self.events[:-1], self.events[1:]):
            left_w, left_ids, left_local, right_w, right_ids, right_local = select_boundary_witnesses(
                candidate_tokens[left.event_id],
                candidate_ids[left.event_id],
                candidate_tokens[right.event_id],
                candidate_ids[right.event_id],
                left.basis,
                right.basis,
                budget=per_boundary,
                boundary_frame=right.start_frame,
            )
            if self.random_witness:
                left_count = candidate_tokens[left.event_id].shape[0]
                combined_tokens = torch.cat([candidate_tokens[left.event_id], candidate_tokens[right.event_id]], dim=0)
                combined_ids = torch.cat([candidate_ids[left.event_id], candidate_ids[right.event_id]], dim=0)
                count = min(per_boundary, combined_tokens.shape[0])
                random_indices = torch.randperm(combined_tokens.shape[0], device=self.device)[:count]
                left_local = random_indices[random_indices < left_count]
                right_local = random_indices[random_indices >= left_count] - left_count
            selected_indices[left.event_id].append(left_local)
            selected_indices[right.event_id].append(right_local)

        for event in self.events:
            parts = [part for part in selected_indices[event.event_id] if part.numel()]
            if parts:
                indices = torch.unique(torch.cat(parts, dim=0), sorted=True)
            else:
                indices = torch.empty((0,), device=self.device, dtype=torch.long)
            tokens = candidate_tokens[event.event_id]
            frame_ids = candidate_ids[event.event_id]
            mask = torch.ones(tokens.shape[0], device=self.device, dtype=torch.bool)
            mask[indices] = False
            event.core_tokens = tokens[mask]
            event.core_frame_ids = frame_ids[mask]
            event.witness_tokens = tokens[indices]
            event.witness_frame_ids = frame_ids[indices]
        if self.misindex_witnesses and len(self.events) > 1:
            witness_banks = [
                (event.witness_tokens, event.witness_frame_ids)
                for event in self.events
            ]
            for event_index, event in enumerate(self.events):
                source_tokens, source_ids = witness_banks[event_index - 1]
                event.witness_tokens = source_tokens
                event.witness_frame_ids = source_ids

    def _oracle_reopen_target_event_id(
        self,
        new_frame_ids: torch.Tensor,
    ) -> Optional[int]:
        if (
            not self.use_oracle_event_target
            or not self.oracle_event_intervals_seconds
            or len(self.events) <= 1
        ):
            return None
        new_timestamps = [
            self.frame_timestamps.get(int(frame_id))
            for frame_id in torch.unique(new_frame_ids, sorted=True).cpu().tolist()
        ]
        new_timestamps = [value for value in new_timestamps if value is not None]
        if not new_timestamps:
            return None
        interval_scores = [
            sum(start <= timestamp <= end for timestamp in new_timestamps)
            for start, end in self.oracle_event_intervals_seconds
        ]
        oracle_index = max(range(len(interval_scores)), key=interval_scores.__getitem__)
        if interval_scores[oracle_index] <= 0:
            return None
        start, end = self.oracle_event_intervals_seconds[oracle_index]
        event_scores = []
        for event in self.events[:-1]:
            _, frame_ids, _ = event.all_tokens()
            timestamps = {
                self.frame_timestamps.get(int(frame_id))
                for frame_id in torch.unique(frame_ids, sorted=True).cpu().tolist()
            }
            score = sum(
                timestamp is not None and start <= timestamp <= end
                for timestamp in timestamps
            )
            event_scores.append((score, event.event_id))
        best_score, best_event_id = max(event_scores, default=(0, -1))
        return best_event_id if best_score > 0 else None

    @torch.no_grad()
    def _strip_post_query_state(self) -> None:
        if self._oracle_sanitized or self.oracle_query_timestamp_seconds is None:
            return
        self._oracle_sanitized = True
        allowed_frame_ids = {
            frame_id
            for frame_id, timestamp in self.frame_timestamps.items()
            if timestamp <= self.oracle_query_timestamp_seconds
        }
        self.oracle_future_frames_used = sum(
            timestamp > self.oracle_query_timestamp_seconds
            for timestamp in self.frame_timestamps.values()
        )
        retained_events = []
        for event in self.events:
            core_mask = torch.tensor(
                [int(frame_id) in allowed_frame_ids for frame_id in event.core_frame_ids.cpu().tolist()],
                device=self.device,
                dtype=torch.bool,
            )
            witness_mask = torch.tensor(
                [int(frame_id) in allowed_frame_ids for frame_id in event.witness_frame_ids.cpu().tolist()],
                device=self.device,
                dtype=torch.bool,
            )
            event.core_tokens = event.core_tokens[core_mask]
            event.core_frame_ids = event.core_frame_ids[core_mask]
            event.witness_tokens = event.witness_tokens[witness_mask]
            event.witness_frame_ids = event.witness_frame_ids[witness_mask]
            tokens, frame_ids, _ = event.all_tokens()
            if not tokens.numel():
                continue
            basis_state = fit_basis(
                tokens,
                min(self.local_rank, self.rank_budget),
            )
            event.basis = basis_state.basis
            event.strengths = basis_state.strengths
            event.intervals = self._intervals(frame_ids)
            residual = residual_scores(tokens, event.basis, relative=True)
            event.residual_mean = (
                float(residual.mean().item()) if residual.numel() else 0.0
            )
            event.confidence = max(0.0, 1.0 - event.residual_mean)
            event.metadata.pop("revision_risk", None)
            retained_events.append(event)
        self.events = retained_events
        self.pel = deque(
            packet for packet in self.pel
            if packet.frame_id in allowed_frame_ids
        )
        self._reset_active_basis()

    @torch.no_grad()
    def _enforce_budget(self) -> None:
        self._prune_pel()
        pel_tokens = sum(frame.tokens.shape[0] for frame in self.pel)
        available_tokens = max(0, self.token_budget - pel_tokens)
        available_rank = max(0, self.rank_budget - self.active_basis_state.rank)
        if not self.events:
            return
        if self.uniform_budget:
            original_risks = []
            token_quotas = [available_tokens // len(self.events)] * len(self.events)
            rank_quotas = [available_rank // len(self.events)] * len(self.events)
            for index in range(available_tokens - sum(token_quotas)):
                token_quotas[index % len(token_quotas)] += 1
            for index in range(available_rank - sum(rank_quotas)):
                rank_quotas[index % len(rank_quotas)] += 1
        else:
            token_quotas, rank_quotas, original_risks = allocate_event_budgets(
                self.events,
                total_tokens=available_tokens,
                total_rank=available_rank,
                min_event_tokens=self.min_event_tokens,
                min_event_rank=self.min_event_rank,
                temperature=self.budget_temperature,
            )
        for event, token_quota, rank_quota in zip(self.events, token_quotas, rank_quotas):
            prune_event(event, token_quota, rank_quota, self.witness_ratio)
        if not self.uniform_budget:
            for event, risk in zip(self.events, original_risks):
                event.metadata["revision_risk"] = risk

    @torch.no_grad()
    def assemble(self, total_num_frames: Optional[int] = None) -> MemoryOutput:
        self._strip_post_query_state()
        self._enforce_budget()
        token_parts: list[torch.Tensor] = []
        id_parts: list[torch.Tensor] = []
        token_types: list[str] = []
        for event in self.events:
            if self.delete_witnesses_at_query:
                tokens = event.core_tokens
                frame_ids = event.core_frame_ids
                types = ["event_core"] * tokens.shape[0]
            else:
                tokens, frame_ids, types = event.all_tokens()
            if tokens.numel():
                token_parts.append(tokens)
                id_parts.append(frame_ids)
                token_types.extend(types)
        for packet in self.pel:
            token_parts.append(packet.tokens)
            id_parts.append(torch.full(
                (packet.tokens.shape[0],), packet.frame_id, device=self.device, dtype=torch.long
            ))
            token_types.extend(["pel"] * packet.tokens.shape[0])

        if not token_parts:
            tokens = torch.empty((0, self.dim), device=self.device, dtype=self.dtype)
            frame_ids = torch.empty((0,), device=self.device, dtype=torch.long)
            sorted_types: list[str] = []
        else:
            tokens = torch.cat(token_parts, dim=0)
            frame_ids = torch.cat(id_parts, dim=0)
            if tokens.shape[0] > self.token_budget:
                tokens = tokens[-self.token_budget:]
                frame_ids = frame_ids[-self.token_budget:]
                token_types = token_types[-self.token_budget:]
            order = torch.argsort(frame_ids, stable=True)
            tokens = tokens[order]
            frame_ids = frame_ids[order]
            order_list = order.cpu().tolist()
            sorted_types = [token_types[index] for index in order_list]

        basis_bytes = (
            self.active_basis_state.basis.numel() * self.active_basis_state.basis.element_size()
            + self.active_basis_state.strengths.numel() * self.active_basis_state.strengths.element_size()
        ) + sum(
            event.basis.numel() * event.basis.element_size()
            + event.strengths.numel() * event.strengths.element_size()
            for event in self.events
        )
        metadata_bytes = sum(
            64 + 16 * len(event.intervals) + sum(len(parent) for parent in event.parent_versions)
            for event in self.events
        )
        frame_token_counts = self._frame_counts(frame_ids, total_num_frames)
        represented_frame_ids = torch.unique(frame_ids, sorted=True).cpu().tolist()
        pel_frame_ids = sorted({int(packet.frame_id) for packet in self.pel})
        return MemoryOutput(
            tokens=tokens,
            frame_ids=frame_ids,
            token_types=sorted_types,
            frame_token_counts=frame_token_counts,
            metadata={
                "method": "revive",
                "visible_tokens": int(tokens.shape[0]),
                "rank_used": int(self.active_basis_state.rank + sum(event.rank for event in self.events)),
                "basis_bytes": int(basis_bytes),
                "metadata_bytes": int(metadata_bytes),
                "pel_tokens": int(sum(kind == "pel" for kind in sorted_types)),
                "pel_frame_ids": pel_frame_ids,
                "witness_tokens": int(sum(kind == "boundary_witness" for kind in sorted_types)),
                "represented_frame_ids": represented_frame_ids,
                "frame_token_counts": frame_token_counts,
                "events": [event.trace_dict() for event in self.events],
                "enable_pel": self.enable_pel,
                "enable_witnesses": self.enable_witnesses,
                "delete_witnesses_at_query": self.delete_witnesses_at_query,
                "misindex_witnesses": self.misindex_witnesses,
                "use_oracle_event_target": self.use_oracle_event_target,
                "oracle_query_timestamp_seconds": self.oracle_query_timestamp_seconds,
                "oracle_future_frames_used": self.oracle_future_frames_used,
            },
        )

