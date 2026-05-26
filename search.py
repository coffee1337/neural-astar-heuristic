"""Forward A* with V(s) as heuristic and batched child evaluation."""

import heapq
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from common import state_key, to_jsonable


def solve_astar(
    state_transition_fn: Callable[[Any, Any], Any],
    action_generator_fn: Callable[[Any], List[Any]],
    heuristic_fn: Optional[Callable[[List[Any]], np.ndarray]],
    initial_state: Any,
    solved_key: str,
    deadline: float,
    max_nodes: int = 50_000,
    expand_batch: int = 32,
) -> Optional[List[str]]:
    """A* with f = g + V(s). Returns action list or None. StateTransitionFn handles state advancement."""
    start = to_jsonable(initial_state)
    start_k = state_key(start)
    if start_k == solved_key:
        return []

    # parents: {state_key: (parent_key, action_taken)} 
    parents: Dict[str, Tuple[Optional[str], Optional[Any]]] = {start_k: (None, None)}
    g_score: Dict[str, int] = {start_k: 0}

    h0 = float(heuristic_fn([start])[0]) if heuristic_fn else 0.0
    counter = 0
    # Heap stores: (f_score, counter, g_score, state_key, state)
    open_heap = [(h0, counter, 0, start_k, start)]
    expanded = 0

    while open_heap:
        if time.time() >= deadline or expanded >= max_nodes:
            return None

        # Pop a batch of nodes to expand.
        batch = []
        while open_heap and len(batch) < expand_batch:
            batch.append(heapq.heappop(open_heap))

        children: List[Tuple[str, Any, str, Any, int]] = [] # (child_key, child_state, parent_key, action, g)
        seen: set[str] = set()

        for _f, _c, g, sk, state in batch:
            if sk == solved_key:
                return _reconstruct(sk, parents)
            try:
                # 1. Get potential actions from the generator function
                actions = action_generator_fn(state)
            except Exception as e:
                print(f"Warning: Error generating actions for state {sk}: {e}")
                continue
            
            for a in actions:
                try:
                    # 2. Transition using the provided function
                    next_state = state_transition_fn(state, a)
                    if next_state is None: continue # Should be handled by fn, but safe check
                    
                    nsk = state_key(next_state)
                except Exception as e:
                    print(f"Warning: Error during state transition for action {a} from {sk}: {e}")
                    continue
                
                ng = g + 1
                if g_score.get(nsk, 1 << 30) <= ng or nsk in seen:
                    continue
                seen.add(nsk)
                children.append((nsk, next_state, sk, a, ng))

        if not children: 
            continue

        # Goal short-circuit on any child.
        for (nsk, _, psk, _, _) in children:
            if nsk == solved_key:
                parents[nsk] = (psk, a) # Note: 'a' here is the action from the last successful child loop iteration, this needs fixing if multiple actions lead to goal.
                # For safety in reconstruction, we must find the correct action that led to nsk
                for _, ns_c, psk_c, a_c, ng_c in children:
                    if state_key(ns_c) == nsk: # Simple check based on assumption of unique paths for goal
                        parents[nsk] = (psk_c, a_c)
                        return _reconstruct(nsk, parents)
                # Fallthrough if complex logic is required to find the exact action. For now, assume first one works.
                parents[nsk] = (psk, a) # Keep original assignment for simplicity after rewrite
                return _reconstruct(nsk, parents)
        
        # Batched V over children.
        child_states = [c[1] for c in children]
        h_vals = heuristic_fn(child_states) if heuristic_fn else np.zeros(len(children), np.float32)

        for (nsk, ns, psk, a, ng), h in zip(children, h_vals):
            parents[nsk] = (psk, a)
            g_score[nsk] = ng
            counter += 1
            heapq.heappush(open_heap, (ng + float(h), counter, ng, nsk, ns))
            expanded += 1
            if expanded >= max_nodes:
                break

    return None


def _reconstruct(end_k: str, parents) -> List[str]:
    actions = []
    cur = end_k
    while True:
        pk, a = parents.get(cur)
        if pk is None or a is None:
            break
        actions.append(a)
        cur = pk
    actions.reverse()
    return actions"""
[END_TOOL_REQUEST]