"""
Shared evaluation helpers for the Pokémon Showdown LLM agent tutorial
(same metrics as `eval_showdown_agent.py`).

Metrics per sample:
  - valid format: output parses as move/switch (extract_cmd)
  - type_match: predicted move vs switch matches ground truth category
  - exact_match: normalized move/switch target string matches ground truth

Used by the standalone eval script and the ROCm tutorial notebook so
short-run and full-checkpoint numbers stay comparable.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Callable, Iterator

SYSTEM_TEMPLATE = (
    "You are a Pokemon Showdown battle AI. You play as {side}. "
    "Given the battle log, output your next action. "
    "Format: move <name> OR switch <name>. "
    "Append terastallize if you terastallize this turn."
)


def postprocess_agent_response(response: str) -> str:
    """Strip Qwen3 thinking wrapper and chat markers (matches notebook inference)."""
    response = response.replace("<|im_end|>", "").strip()
    response = re.sub(
        r"(?s)^<think>.*?</think>\s*", "", response
    ).strip()
    return response


def showdown_fields(line: str) -> list[str]:
    """
    Split a Pokemon Showdown protocol line into fields.

    Lines usually look like ``|player|p1|Name|...``. A plain ``split('|')`` leaves
    leading empty segments; we strip those so the message type is always ``fields[0]``.
    """
    parts = line.strip().split("|")
    while parts and parts[0] == "":
        parts = parts[1:]
    return parts


# --- Tutorial / single-turn legality ---


def parse_showdown_action_line(line: str) -> dict[str, Any] | None:
    """First line: 'move Name ...' / 'switch Name ...' with optional trailing 'terastallize'."""
    line = line.strip()
    if not line:
        return None
    m = re.match(r"^(move|switch)\s+(.+)$", line, flags=re.IGNORECASE)
    if not m:
        return None
    verb = m.group(1).lower()
    rest = m.group(2).strip()
    tera = False
    low = rest.lower()
    if low.endswith(" terastallize"):
        tera = True
        rest = rest[: -len(" terastallize")].strip()
    if not rest:
        return None
    return {"verb": verb, "target": rest, "tera": tera}


def extract_last_request_json(log_text: str) -> dict[str, Any] | None:
    """Parse the last |request|{...} line (Showdown turn JSON)."""
    for raw in reversed(log_text.strip().splitlines()):
        raw = raw.strip()
        if not raw.startswith("|request|"):
            continue
        rest = raw[len("|request|") :]
        if rest.startswith("{"):
            blob = rest
        else:
            idx = rest.find("{")
            if idx == -1:
                continue
            blob = rest[idx:]
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    return None


def species_from_request_pokemon(p: dict[str, Any]) -> str:
    det = (p.get("details") or "").strip()
    if det:
        return det.split(",")[0].strip()
    ident = (p.get("ident") or "").strip()
    if ":" in ident:
        return ident.split(":", 1)[1].strip()
    return ""


def legal_moves_from_request(req: dict[str, Any]) -> list[str]:
    names: list[str] = []
    if not req or "active" not in req:
        return names
    for slot in req.get("active") or []:
        for m in slot.get("moves") or []:
            if m.get("disabled"):
                continue
            mv = m.get("move")
            if mv:
                names.append(mv.casefold())
    return names


def legal_switch_species_from_request(req: dict[str, Any]) -> list[str]:
    out: list[str] = []
    side = (req or {}).get("side") or {}
    for p in side.get("pokemon") or []:
        cond = (p.get("condition") or "").lower()
        if "fnt" in cond or cond.startswith("0 "):
            continue
        if p.get("active"):
            continue
        sp = species_from_request_pokemon(p)
        if sp:
            out.append(sp.casefold())
    return out


def can_terastallize_from_request(req: dict[str, Any]) -> bool:
    if not req or "active" not in req:
        return False
    for slot in req.get("active") or []:
        if slot.get("canTerastallize"):
            return True
    return False


def p2_roster_and_active_from_log(log_text: str) -> tuple[list[str], str | None]:
    """Species names from |switch|p2*| lines; active p2a from the last such line."""
    roster: list[str] = []
    active_p2a: str | None = None
    for raw in log_text.splitlines():
        f = showdown_fields(raw)
        if len(f) < 3 or f[0] != "switch":
            continue
        slot = f[1]
        if not re.match(r"p2[a-z]?:", slot):
            continue
        species = f[2].split(",")[0].strip()
        if not species:
            continue
        roster.append(species)
        if slot.startswith("p2a:"):
            active_p2a = species
    return roster, active_p2a


def validate_action_against_log(action_line: str, log_text: str) -> dict[str, Any]:
    """
    Structure + optional |request| legality. Keys are stable for notebook printing.
    """
    roster, active = p2_roster_and_active_from_log(log_text)
    roster_set = {s.casefold() for s in roster}
    req = extract_last_request_json(log_text)
    parsed = parse_showdown_action_line(action_line)
    if not parsed:
        return {
            "parsed": None,
            "structure_ok": False,
            "single_line_ok": False,
            "request_present": req is not None,
            "move_legal_in_request": None,
            "switch_legal_in_request": None,
            "tera_legal_in_request": None,
            "switch_target_ok": None,
            "move_name_nonempty": None,
            "notes": "does not match move/switch line grammar",
        }
    lines = [ln for ln in action_line.splitlines() if ln.strip()]
    single = len(lines) <= 1
    mv_ok = None
    sw_ok = None
    move_legal = None
    switch_legal = None
    tera_legal = None
    if parsed["verb"] == "move":
        mv_ok = bool(parsed["target"].strip())
        if req:
            legal = set(legal_moves_from_request(req))
            move_legal = parsed["target"].casefold() in legal
            if parsed["tera"]:
                tera_legal = can_terastallize_from_request(req)
            else:
                tera_legal = True
    if parsed["verb"] == "switch":
        tgt = parsed["target"].casefold()
        sw_ok = tgt in roster_set and (active is None or tgt != active.casefold())
        if req:
            legal_sw = set(legal_switch_species_from_request(req))
            switch_legal = tgt in legal_sw
    return {
        "parsed": parsed,
        "structure_ok": True,
        "single_line_ok": single,
        "request_present": req is not None,
        "move_legal_in_request": move_legal,
        "switch_legal_in_request": switch_legal,
        "tera_legal_in_request": tera_legal,
        "switch_target_ok": sw_ok,
        "move_name_nonempty": mv_ok,
        "notes": "ok",
    }


def tutorial_demo_log_with_request() -> str:
    """
    Fixed battle prefix + synthetic |request| so tutorials can check move legality.
    Corviknight four moves are OU-plausible; Earthquake is intentionally NOT listed.
    """
    req = {
        "active": [
            {
                "moves": [
                    {"move": "Brave Bird", "id": "bravebird", "pp": 24, "maxpp": 24, "disabled": False},
                    {"move": "Iron Head", "id": "ironhead", "pp": 24, "maxpp": 24, "disabled": False},
                    {"move": "Roost", "id": "roost", "pp": 8, "maxpp": 8, "disabled": False},
                    {"move": "U-turn", "id": "uturn", "pp": 24, "maxpp": 24, "disabled": False},
                ],
                "canTerastallize": "Flying",
            }
        ],
        "side": {
            "pokemon": [
                {"ident": "p2: Corviknight", "details": "Corviknight, M, L50", "condition": "100/100", "active": True},
                {"ident": "p2: Dragapult", "details": "Dragapult, M, L50", "condition": "100/100", "active": False},
            ]
        },
    }
    return (
        "|player|p1|Player1|266|1500\n"
        "|player|p2|Player2|1|1500\n"
        "|teamsize|p1|6\n"
        "|teamsize|p2|6\n"
        "|gen|9\n"
        "|tier|[Gen 9] OU\n"
        "|\n"
        "|start\n"
        "|switch|p1a: Garchomp|Garchomp, M|100/100\n"
        "|switch|p2a: Corviknight|Corviknight, M|100/100\n"
        "|turn|1\n"
        "|move|p1a: Garchomp|Earthquake|p2a: Corviknight\n"
        "|-immune|p2a: Corviknight\n"
        "|request|"
        + json.dumps(req, separators=(",", ":"))
        + "\n|turn|2"
    )


def extract_winner_side(log_text: str):
    winner = None
    players = {}
    for line in log_text.split("\n"):
        f = showdown_fields(line)
        if len(f) >= 3 and f[0] == "player":
            players[f[1]] = f[2]
        if len(f) >= 2 and f[0] == "win":
            winner = f[1]

    if not winner:
        return None, None

    for side, name in players.items():
        if name == winner:
            return side, winner
    return None, winner


def build_test_samples(
    dataset: Iterator[dict[str, Any]],
    num_samples: int,
    *,
    seed: int = 42,
    min_rating: int = 1400,
    require_gen9: bool = True,
    max_scans: int = 400_000,
) -> list[dict[str, Any]]:
    """
    Build (prompt, gt_action, side, ...) tuples using the same logic as
    `eval_showdown_agent.py`: winner's action at a random mid-game turn.
    """
    random.seed(seed)
    samples: list[dict[str, Any]] = []
    scanned = 0

    for row in dataset:
        scanned += 1
        if len(samples) >= num_samples:
            break
        if scanned > max_scans:
            break

        fmt = (row.get("formatid") or "").lower().replace(" ", "")
        if require_gen9 and "gen9" not in fmt:
            continue
        if (row.get("rating") or 0) < min_rating:
            continue

        log_text = row["log"]
        winner_side, _winner_name = extract_winner_side(log_text)
        if not winner_side:
            continue

        lines = log_text.strip().split("\n")

        turn_positions: list[tuple[int, int]] = []
        for i, line in enumerate(lines):
            f = showdown_fields(line)
            if len(f) >= 2 and f[0] == "turn":
                try:
                    turn_positions.append((int(f[1]), i))
                except ValueError:
                    pass

        if not turn_positions or len(turn_positions) <= 3:
            continue

        target_turn_idx = random.randint(2, len(turn_positions) - 2)
        _turn_num, turn_line_idx = turn_positions[target_turn_idx]

        if target_turn_idx + 1 < len(turn_positions):
            next_turn_line = turn_positions[target_turn_idx + 1][1]
        else:
            next_turn_line = len(lines)

        gt_action = None
        for j in range(turn_line_idx + 1, next_turn_line):
            f = showdown_fields(lines[j])
            if len(f) < 3:
                continue
            if f[0] == "move" and f[1].startswith(f"{winner_side}a:"):
                tera = ""
                start_look = max(0, j - 3)
                end_look = min(len(lines), j + 3)
                if any(
                    "terastallize" in lines[k] and winner_side in lines[k]
                    for k in range(start_look, end_look)
                ):
                    tera = " terastallize"
                gt_action = f"move {f[2]}{tera}"
                break
            if f[0] == "switch" and f[1].startswith(f"{winner_side}a:"):
                pokemon = f[1].split(": ", 1)[1] if ": " in f[1] else f[1]
                gt_action = f"switch {pokemon}"
                break

        if not gt_action:
            continue

        log_prefix = "\n".join(lines[: turn_line_idx + 1])
        samples.append(
            {
                "rating": row["rating"],
                "format": row["formatid"],
                "side": winner_side,
                "prompt": log_prefix,
                "gt_action": gt_action,
            }
        )

    return samples


def extract_cmd(text: str):
    """Extract 'move X' or 'switch X' from model output (eval script regex)."""
    m = re.search(
        r"(move\s+[\w\-]+(?:\s+[\w\-]+)?|switch\s+[\w\-]+(?:\s+[\w\-]+)?)",
        text,
        re.IGNORECASE,
    )
    if m:
        cmd = m.group(1).strip()
        cmd_type = "move" if cmd.lower().startswith("move") else "switch"
        return cmd, cmd_type
    return None, None


def eval_showdown_agent_batch(
    model,
    tokenizer,
    samples: list[dict[str, Any]],
    *,
    max_new_tokens: int = 30,
    use_cache: bool = False,
    device: str = "cuda",
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """
    Run inference on prepared samples; return aggregate metrics and per-row details.
    Prompting uses `apply_chat_template` to match the training notebook.
    """
    import torch

    metrics = {
        "total": 0,
        "move_pred": 0,
        "switch_pred": 0,
        "invalid_pred": 0,
        "type_match": 0,
        "exact_match": 0,
    }
    rows: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_TEMPLATE.format(side=sample["side"]),
            },
            {"role": "user", "content": sample["prompt"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=use_cache,
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "<|im_start|>assistant\n" in full_response:
            response = full_response.split("<|im_start|>assistant\n")[-1]
        else:
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
            )
        response = postprocess_agent_response(response)
        elapsed = time.time() - t0

        pred_cmd, pred_type = extract_cmd(response)
        gt_cmd, gt_type = extract_cmd(sample["gt_action"])

        metrics["total"] += 1
        if pred_type == "move":
            metrics["move_pred"] += 1
        elif pred_type == "switch":
            metrics["switch_pred"] += 1
        else:
            metrics["invalid_pred"] += 1

        type_match = (pred_type == gt_type) if (pred_type and gt_type) else False
        exact_match = False
        if pred_cmd and gt_cmd:
            p_name = " ".join(pred_cmd.lower().split()[1:])
            g_name = " ".join(gt_cmd.lower().split()[1:])
            if p_name == g_name:
                exact_match = True

        if type_match:
            metrics["type_match"] += 1
        if exact_match:
            metrics["exact_match"] += 1

        row = {
            "i": i,
            "rating": sample["rating"],
            "format": sample["format"],
            "gt_action": sample["gt_action"],
            "response_head": response[:500],
            "pred_cmd": pred_cmd,
            "pred_type": pred_type,
            "type_match": type_match,
            "exact_match": exact_match,
            "elapsed_s": elapsed,
        }
        rows.append(row)
        if progress:
            progress(i + 1, len(samples), row)

    return metrics, rows


def print_metrics_summary(metrics: dict[str, int], title: str = "Evaluation summary") -> None:
    t = metrics["total"]
    if t == 0:
        print(f"{title}: no samples")
        return
    valid = metrics["move_pred"] + metrics["switch_pred"]
    print(f"{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    print(f"Total samples: {t}")
    print(
        f"Valid-format predictions: {valid}/{t} ({100.0 * valid / t:.1f}%) "
        f"(move={metrics['move_pred']}, switch={metrics['switch_pred']}, invalid={metrics['invalid_pred']})"
    )
    print(
        f"Type match (move vs switch): {metrics['type_match']}/{t} ({100.0 * metrics['type_match'] / t:.1f}%)"
    )
    print(
        f"Exact match (full action): {metrics['exact_match']}/{t} ({100.0 * metrics['exact_match'] / t:.1f}%)"
    )
    print(f"{'=' * 60}")


def save_metrics_json(
    path: str,
    metrics: dict[str, int],
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    out = dict(metrics)
    if extra:
        out["meta"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
