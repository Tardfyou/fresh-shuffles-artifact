"""Order Recovery Experiment v2: Full Protocol Implementation

Implements the experimental protocol from KVCLOAK_ORDER_RECOVERY_PROTOCOL.md
- 600 main samples (3 categories × 4 lengths × 50)
- 60 development samples
- 100 "same multiset, different fact" pairs (boundary test A)
- 100 random sequences (boundary test B)
- 5 controlled conditions per sample
- Proper metrics with clustered bootstrap CI
- Full artifact saving (JSONL files)

Usage:
  python exp_order_v2.py --phase smoke    # 3 samples, verify pipeline
  python exp_order_v2.py --phase dev      # 60 dev samples
  python exp_order_v2.py --phase main     # 600 main samples (frozen)
  python exp_order_v2.py --phase boundary # 200 boundary samples
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict

import torch

# ============================================================
# Configuration (frozen before main run)
# ============================================================
CONFIG = {
    "model": "Qwen/Qwen2.5-0.5B",
    "block_size": 16,
    "beam_width": 16,
    "top_k": 1,          # main result is top-1 only
    "context_limit": 96,  # max context tokens for beam search
    "n_iters": 2,         # iterative refinement passes
    "seed_calibration": 42,
    "seed_obfuscation_base": 9000,
    "seed_sampling": 20260929,
    "device": "cpu",
    "threads": 16,
}

PHASES = {"smoke": 3, "dev": 60, "main": 600, "boundary": 200}
# Results go to the HDD pool (crash-safe against SSD transient full-disk events;
# also avoids wearing the SSD with large experimental data).
OUTDIR = "/mnt/hdd/research/kvcloak_results/order_v2"


# ============================================================
# Data Generation
# ============================================================
rng = random.Random(CONFIG["seed_sampling"])

# --- Natural conversation templates (varied, no type keywords) ---
CONVO_TEMPLATES = [
    "I think we should {action} tonight at that {adj} place on {street}.",
    "Let me know when you're {state} to {action} and I'll {helper}.",
    "She told me that the {thing} was already {state} when she {action_past}.",
    "Can you believe the {thing} actually {action_past} without any {helper}?",
    "We were planning to {action} but then the {thing} started {state}.",
    "After the {event} everyone wanted to {action} but only {group} could go.",
    "The {person} mentioned that {thing} might be {state} for the {event}.",
    "I heard from {person} that there's a new {thing} opening near {street}.",
    "They decided to {action} instead of {alt_action} because of the {thing}.",
    "It's been {duration} since we last {action_past} together at {place}.",
    "The {group} agreed that {thing} was the best {category} they'd seen.",
    "Remember when {person} tried to {action} and ended up {state}?",
    "If you want to {action} you should probably {helper} first.",
    "Most people don't realize how much {thing} can affect your {category}.",
    "The key is to {action} before the {thing} becomes too {adj} to handle.",
    "What surprised me most was how quickly the {thing} went from {state_a} to {state_b}.",
    "She spent the entire {duration} trying to convince {person} to {action}.",
    "Nobody expected the {thing} to be so {adj} given its {category}.",
    "He kept saying he would {action} but somehow never found the {helper}.",
    "The {event} turned out completely different from what {person} had predicted.",
]

CONVO_FILLS = {
    "action": ["grab dinner", "watch a movie", "go hiking", "meet up", "check out the exhibit",
               "try that recipe", "take a class", "visit the museum", "see the show", "go swimming"],
    "adj": ["new", "interesting", "quiet", "popular", "charming", "unusual", "remarkable", "hidden"],
    "street": ["Main Street", "Oak Avenue", "River Road", "Elm Court", "Park Lane", "Hill Boulevard"],
    "state": ["ready", "available", "finished", "too crowded", "closed", "already done", "prepared"],
    "thing": ["book", "restaurant", "presentation", "proposal", "exhibit", "concert", "project", "recipe"],
    "helper": ["call ahead", "make a reservation", "check the schedule", "ask permission", "bring supplies"],
    "person": ["Sarah", "Mike", "the manager", "my colleague", "her friend", "his neighbor"],
    "action_past": ["arrived", "finished it", "left early", "changed plans", "found it", "signed up"],
    "group": ["a few people", "the team", "everyone in the class", "two of us", "the whole family"],
    "event": ["meeting", "conference", "workshop", "wedding", "graduation", "holiday party"],
    "alt_action": ["stay home", "postpone it", "do something else", "take a break"],
    "duration": ["three weeks", "several months", "almost a year", "just two days"],
    "place": ["the office", "that café", "her apartment", "the library", "downtown"],
    "category": ["one they'd tried", "option available", "version released this year"],
    "state_a": ["completely unknown", "barely usable", "just an idea"],
    "state_b": ["everywhere", "essential", "a household name"],
}

# --- Narrative/explanatory templates ---
NARR_TEMPLATES = [
    "The {subject} had always {past_action}. After {time_period} she finally {resolution}.",
    "During the {season} our {group} traveled to {location} to study {topic}.",
    "The {field} report showed that {phenomenon} had {change} over the past {period}.",
    "When the {event} hit the {place}, residents rushed to {response} while {group} prepared {preparation}.",
    "After graduating from {institution}, she moved to {city} to pursue {career}.",
    "The {process} begins when {trigger} causes {effect}, leading to {outcome}.",
    "In the {field} literature, {concept} refers to the {definition} observed in {context}.",
    "The discovery that {finding} challenged the prevailing assumption that {assumption}.",
]

NARR_FILLS = {
    "subject": ["young woman", "research scientist", "aspiring musician", "graduate student"],
    "past_action": ["dreamed of living abroad", "wanted to study art", "hoped to write a novel"],
    "time_period": ["saving for three years", "months of preparation", "a full semester of planning"],
    "resolution": ["bought a plane ticket", "submitted her application", "enrolled in the program"],
    "season": ["summer", "autumn", "spring break"],
    "group": ["research team", "study group", "expedition"],
    "location": ["the mountains", "a remote island", "the ancient city"],
    "topic": ["local wildlife", "traditional customs", "geological formations"],
    "field": ["economic", "climate", "medical", "environmental"],
    "phenomenon": ["inflation", "temperature variation", "sea level", "crop yield"],
    "change": ["decreased significantly", "increased steadily", "remained stable"],
    "period": ["quarter", "decade", "five years"],
    "event": ["storm", "flood", "wildfire", "heat wave"],
    "place": ["coastal town", "river valley", "mountain village"],
    "response": ["secure their homes", "evacuate to shelters", "gather supplies"],
    "group2": ["emergency services", "local volunteers", "the national guard"],
    "preparation": ["temporary housing", "medical stations", "communication networks"],
    "institution": ["college", "university", "trade school"],
    "city": ["New York", "Chicago", "San Francisco"],
    "career": ["acting", "journalism", "architecture", "research"],
    "process": ["inflammation response", "crystallization", "natural selection"],
    "trigger": ["tissue damage", "temperature drop", "environmental pressure"],
    "effect": ["immune activation", "molecular rearrangement", "trait selection"],
    "outcome": ["healing", "structure formation", "adaptation"],
    "concept": ["cognitive dissonance", "emergence", "homeostasis"],
    "definition": ["psychological discomfort", "complex behavior from simple rules", "self-regulation"],
    "context": ["decision-making", "ant colonies", "biological systems"],
    "finding": ["memory is reconstructive", "particles can be entangled", "gut bacteria affect mood"],
    "assumption": ["memory works like a recording", "particles are independent", "the brain is isolated"],
}

# --- Structured/synthetic templates (fictional data) ---
STRUCT_TEMPLATES = [
    "My name is {name} and I can be reached at {email} or {phone}.",
    "The account {account_num} has a current balance of {amount} as of today.",
    "Please ship the package to {address} using {shipping} with tracking.",
    "The server {hostname} is running {software} on port {port} with SSL enabled.",
    "For verification purposes my employee ID is {emp_id} in the {dept} department.",
    "The reservation is under {name} for {party_size} people on {date} at {time}.",
    "Your confirmation code is {code} and the total charge is {amount} to card ending {last_four}.",
]

STRUCT_FILLS = {
    "name": ["Jordan Lee", "Casey Wong", "Riley Chen", "Avery Kim", "Quinn Park",
             "Morgan Reid", "Blake Foster", "Sage Nakamura", "Reese Vance", "Kai Dubois"],
    "email": ["jordan.lee@mail-example.com", "casey.w@sample-domain.org", "riley.chen@test-example.net"],
    "phone": ["555-214-8307", "555-918-4263", "555-637-2091"],
    "account_num": ["XX-4821", "XX-7309", "XX-1546"],
    "amount": ["$1,247.83", "$892.16", "$3,406.99"],
    "address": ["742 Evergreen Terrace, Springfield", "118 Pinehurst Lane, Portland"],
    "shipping": ["priority overnight", "two-day express", "ground service"],
    "hostname": ["srv-prod-04.example.internal", "api-gateway-02.example.io"],
    "software": ["nginx 1.24", "Apache 2.4", "Caddy 2.7"],
    "port": ["8443", "9090", "3000"],
    "emp_id": ["EMP-2024-8391", "EMP-2023-4502", "EMP-2025-1173"],
    "dept": ["Engineering", "Operations", "Product"],
    "party_size": ["four", "six", "two"],
    "date": ["March 15th", "April 22nd", "June 8th"],
    "time": ["7:30 PM", "noon", "3:45 PM"],
    "code": ["KX7M-92PL", "TR4N-88WQ", "ZF2B-61HD"],
    "last_four": ["4578", "8321", "0946"],
}


def fill_template(template, fills):
    """Fill a template with random values from the fills dictionary."""
    import re
    def replacer(match):
        key = match.group(1)
        if key in fills:
            return rng.choice(fills[key])
        return match.group(0)
    return re.sub(r'\{(\w+)\}', replacer, template)


def generate_sample(category):
    """Generate one sentence for a given category."""
    if category == "conversation":
        tmpl = rng.choice(CONVO_TEMPLATES)
        return fill_template(tmpl, CONVO_FILLS)
    elif category == "narrative":
        tmpl = rng.choice(NARR_TEMPLATES)
        fills = {**NARR_FILLS}
        fills["group2"] = fills.get("group2", NARR_FILLS.get("group", ["the team"]))
        return fill_template(tmpl, fills)
    elif category == "structured":
        tmpl = rng.choice(STRUCT_TEMPLATES)
        return fill_template(tmpl, STRUCT_FILLS)
    else:
        raise ValueError(f"Unknown category: {category}")


def generate_sample_of_length(category, target_blocks, tok):
    """Generate text spanning exactly target_blocks complete blocks (target_blocks*B tokens).

    Concatenates independently-filled template sentences until the tokenized
    length reaches target_blocks*B, then truncates to the block boundary.
    Guarantees nb >= target_blocks (never returns a too-short sample).
    """
    B = CONFIG["block_size"]
    need = target_blocks * B
    parts = []
    total = 0
    guard = 0
    while total < need and guard < 50:
        sent = generate_sample(category)
        parts.append(sent)
        total = len(tok(" ".join(parts), return_tensors="pt").input_ids[0])
        guard += 1
    full_text = " ".join(parts)
    ntok = len(tok(full_text, return_tensors="pt").input_ids[0])
    if ntok < need:
        # extremely unlikely after guard; raise rather than silently skip
        raise RuntimeError(f"could not reach {need} tokens for {category} "
                           f"(got {ntok} after {guard} sentences)")
    return full_text


def generate_boundary_pair():
    """Generate a pair with same multiset but different facts (direction reversal).
    v2: wrapped in a shared neutral context so tokenized length >= one block
    (run2's bare sentences were all skipped_too_short). Wrap is IDENTICAL for
    both directions, preserving the multiset-identity constraint."""
    actors = rng.sample(["Alex", "Bailey", "Casey", "Drew", "Emery", "Finley"], 2)
    amount = rng.choice(["fifty dollars", "two hundred dollars", "seventy-five dollars"])
    method = rng.choice(["bank transfer", "mobile payment", "wire transfer"])
    PRE = ("Financial transaction record for compliance review, filed this quarter. "
           "Verified entry: ")
    POST = (" The compliance department has acknowledged and archived this entry.")
    text_a = f"{PRE}{actors[0]} sent {amount} to {actors[1]} via {method} yesterday afternoon.{POST}"
    text_b = f"{PRE}{actors[1]} sent {amount} to {actors[0]} via {method} yesterday afternoon.{POST}"
    return text_a, text_b


def generate_random_sequence(min_tokens=16, max_tokens=48):
    """Generate a random token sequence with no linguistic structure."""
    length = rng.randint(min_tokens, max_tokens)
    # Use random token IDs from a mid-range (avoid special tokens and very rare ones)
    ids = [rng.randint(1000, 50000) for _ in range(length)]
    return ids


# ============================================================
# Attack Pipeline
# ============================================================
def build_attack():
    """Build the full attack pipeline (model + calibration + dictionary)."""
    from attack_lib import (load_model, model_dims, probe_ids, build_theta,
                            build_dictionary, cluster_states, recover_secrets,
                            attack_block)
    from run_suite import NATURAL
    from get_kvcloak_config import get_kvcloak_config
    from kvcloak import KVCloak

    torch.manual_seed(CONFIG["seed_calibration"])
    B = CONFIG["block_size"]

    tok, model = load_model(CONFIG["model"], threads=CONFIG["threads"])
    layers, H, D = model_dims(model)

    theta = build_theta(model, tok, NATURAL)
    torch.manual_seed(CONFIG["seed_calibration"])
    cfg = get_kvcloak_config(layers, H, D, B, theta, 1.0, 1.0, 2.0)
    cloak = KVCloak(cfg, dtype=torch.bfloat16, fused=False, need_ratio=False, add_a=True)

    # Calibrate all heads
    pids, prefix_len, _ = probe_ids(tok, target=2048,
                                    max_positions=getattr(model.config, "max_position_embeddings", None))
    with torch.no_grad():
        pc = model(input_ids=pids, use_cache=True).past_key_values
    NB = pids.shape[1] // B
    first_blk = (prefix_len + B - 1) // B
    plain_probe = pc[0][1][0, :, first_blk*B:NB*B].clone()
    torch.manual_seed(CONFIG["seed_calibration"])
    prot = cloak.obfuscate(pc)
    nblk_pad = prot[0][1].shape[2] // B

    all_secs = {}
    for hh in range(H):
        bc = prot[0][1][0, hh].float().view(nblk_pad, B, D)[first_blk:NB]
        ci, ce, *_ = cluster_states(bc)
        if ci == B:
            all_secs[hh] = recover_secrets(ce, plain_probe[hh], B)

    DICT, _, _ = build_dictionary(model, H, D)
    Dn = torch.nn.functional.normalize(DICT, dim=-1)
    bos = tok.bos_token_id or tok.eos_token_id

    return {
        "tok": tok, "model": model, "cloak": cloak,
        "all_secs": all_secs, "Dn": Dn, "bos": bos,
        "H": H, "D": D, "B": B,
    }


def recover_multiset(atk, pv, nb):
    """Recover per-block multisets using all calibrated heads (majority vote)."""
    from attack_lib import attack_block
    B = atk["B"]
    block_ms = []
    for k in range(nb):
        head_preds = []
        for hh, s in atk["all_secs"].items():
            C = pv[0][1][0, hh].float()[k*B:(k+1)*B]
            preds = attack_block(C, s["s"], s["a_hat"], s["M"], atk["Dn"][:, hh])
            head_preds.append(preds)
        # Majority vote per position (ensures exactly B tokens per block)
        fused = []
        for pos in range(B):
            votes = Counter(hp[pos] for hp in head_preds if pos < len(hp))
            fused.append(votes.most_common(1)[0][0])
        block_ms.append(Counter(fused))
    return block_ms


def beam_unscramble(atk, ms_input, context_ids):
    """Constraint beam search. Input is a Counter, output is an ordered list."""
    model = atk["model"]
    width = CONFIG["beam_width"]
    ms = dict(ms_input)
    beams = [(0.0, list(context_ids), ms)]
    total = sum(ms.values())
    ctx_limit = CONFIG["context_limit"]

    for step in range(total):
        candidates = []
        for score, seq, mx in beams:
            if not mx:
                candidates.append((score, seq, mx)); continue
            ctx = seq[-ctx_limit:]
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ctx])).logits[0, -1]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            for t in list(mx.keys()):
                if mx[t] <= 0: continue
                m2 = dict(mx); m2[t] -= 1
                if m2[t] == 0: del m2[t]
                candidates.append((score + log_probs[t].item(), seq + [t], m2))

        # Deduplicate by recent 12 tokens
        seen = {}
        for sc, sq, mx in candidates:
            key = tuple(sq[-12:])
            if key not in seen or sc > seen[key][0]:
                seen[key] = (sc, sq, mx)
        beams = sorted(seen.values(), key=lambda x: -x[0])[:width]

    return beams[0][1][len(context_ids):]


def order_recovery(atk, block_ms, nb, use_cross_block=True):
    """Sequential beam ordering with iterative refinement."""
    bos = atk["bos"]
    ctx_limit = CONFIG["context_limit"]

    best_ordered = None
    for it in range(CONFIG["n_iters"]):
        ctx = [bos]
        ordered = []
        for blk in range(nb):
            if use_cross_block:
                c = ctx[-ctx_limit:]
            else:
                c = [bos]
            od = beam_unscramble(atk, block_ms[blk], c)
            ctx = (ctx + od)[-ctx_limit:]
            ordered.extend(od)
        best_ordered = ordered
    return best_ordered


# ============================================================
# Evaluation
# ============================================================
def evaluate(true_tokens, predicted_tokens, block_ms, B=16):
    """Compute all metrics for one sample."""
    true = list(true_tokens)
    pred = list(predicted_tokens)

    # Length alignment
    n = min(len(true), len(pred))

    # Position accuracy
    pos_correct = sum(1 for i in range(n) if true[i] == pred[i])
    pa = pos_correct / max(1, len(true))

    # Exact match (full evaluation span)
    exact = (true == pred)

    # Block-level exact match
    nb = len(true) // B
    block_exact = sum(1 for k in range(nb)
                     if true[k*B:(k+1)*B] == pred[k*B:(k+1)*B]) if nb > 0 else 0

    # Trigram overlap
    tri_n = 3
    if len(pred) >= tri_n and len(true) >= tri_n:
        pred_grams = set(tuple(pred[i:i+tri_n]) for i in range(len(pred)-tri_n+1))
        true_grams = set(tuple(true[i:i+tri_n]) for i in range(len(true)-tri_n+1))
        tri = len(pred_grams & true_grams) / max(1, len(true_grams))
    else:
        tri = 0.0

    # Multiset conservation check
    total_ms = Counter()
    for ms in block_ms:
        total_ms += ms
    pred_ms = Counter(pred)
    ms_match = sum((total_ms & pred_ms).values()) / max(1, sum(total_ms.values()))

    # Position accuracy by block index
    pa_by_block = []
    for k in range(nb):
        tc = sum(1 for i in range(k*B, min((k+1)*B, n)) if true[i] == pred[i])
        block_len = min((k+1)*B, n) - k*B
        pa_by_block.append(round(tc / max(1, block_len), 4))

    return {
        "position_accuracy": round(pa, 4),
        "exact_match": exact,
        "block_exact": block_exact,
        "block_total": nb,
        "trigram_overlap": round(tri, 4),
        "multiset_conservation": round(ms_match, 4),
        "pa_by_block": pa_by_block,
        "n_true_tokens": len(true),
        "n_pred_tokens": len(pred),
    }


def random_baseline_pa(true_tokens, B=16):
    """Expected PA under uniform random permutation of the correct multiset."""
    true = list(true_tokens)
    counts = Counter(true)
    n = len(true)
    return sum(c**2 for c in counts.values()) / (n**2)


# ============================================================
# Main Experiment Loop
# ============================================================
def run_experiment(atk, samples, phase_name, conditions=None, sinks=None, skip_ids=None):
    """Run all conditions for a list of samples.
    sinks: optional dict of opened files {predictions, metrics, costs} — when given,
    each sample's record is written and flushed IMMEDIATELY (crash-safe: a disk-full
    event loses at most one sample, never the whole run)."""
    tok = atk["tok"]
    model = atk["model"]
    cloak = atk["cloak"]
    B = atk["B"]

    if conditions is None:
        conditions = ["end2end", "random_order", "oracle_ms", "no_cross_block"]

    def _emit(rec, which):
        if sinks and sinks.get(which):
            sinks[which].write(json.dumps(rec, ensure_ascii=False) + "\n")
            sinks[which].flush()

    all_results = []
    all_metrics = []
    all_costs = []

    for si, sample in enumerate(samples):
        sample_id = sample["id"]
        if skip_ids and sample_id in skip_ids:
            continue
        text = sample["text"]
        category = sample.get("category", "unknown")

        # Tokenize
        if isinstance(text, list):
            vids = torch.tensor([text])  # already token IDs (random sequence)
        else:
            vids = tok(text, return_tensors="pt").input_ids

        nb = vids.shape[1] // B
        if nb == 0:
            rec = {"id": sample_id, "status": "skipped_too_short"}
            all_results.append(rec)
            _emit(rec, "predictions")
            continue

        vids = vids[:, :nb * B]
        true_tokens = vids[0].tolist()

        # Obfuscate (service side)
        t0 = time.time()
        with torch.no_grad():
            vc = model(input_ids=vids, use_cache=True).past_key_values
        torch.manual_seed(CONFIG["seed_obfuscation_base"] + si)
        pv = cloak.obfuscate(vc)
        t_obf = time.time() - t0

        # Recover multiset (attacker)
        t0 = time.time()
        block_ms = recover_multiset(atk, pv, nb)
        t_ms = time.time() - t0

        sample_results = {"id": sample_id, "category": category, "blocks": nb,
                         "tokens": len(true_tokens)}
        sample_metrics = {"id": sample_id, "category": category}
        sample_costs = {"id": sample_id, "obf_s": round(t_obf, 3),
                       "multiset_s": round(t_ms, 3)}

        for cond in conditions:
            t0 = time.time()
            if cond == "end2end":
                # Full pipeline: recovered multiset + beam + cross-block context
                ordered = order_recovery(atk, block_ms, nb, use_cross_block=True)
            elif cond == "random_order":
                # Random permutation within each block
                ordered = []
                for ms in block_ms:
                    tokens = []
                    for t, c in ms.items():
                        tokens.extend([t] * c)
                    rng.shuffle(tokens)
                    ordered.extend(tokens)
            elif cond == "oracle_ms":
                # Correct multiset (from true tokens) + same beam search
                oracle_ms = [Counter(true_tokens[k*B:(k+1)*B]) for k in range(nb)]
                ordered = order_recovery(atk, oracle_ms, nb, use_cross_block=True)
            elif cond == "no_cross_block":
                # Each block independently, no context from previous blocks
                ordered = order_recovery(atk, block_ms, nb, use_cross_block=False)
            else:
                continue

            t_cond = time.time() - t0

            metrics = evaluate(true_tokens, ordered, block_ms, B)
            metrics["condition"] = cond
            metrics["time_s"] = round(t_cond, 2)
            metrics["random_baseline_pa"] = round(random_baseline_pa(true_tokens, B), 4)

            sample_metrics[cond] = metrics
            sample_costs[f"{cond}_s"] = round(t_cond, 2)

        all_results.append(sample_results)
        all_metrics.append(sample_metrics)
        all_costs.append(sample_costs)
        _emit(sample_results, "predictions")
        _emit(sample_metrics, "metrics")
        _emit(sample_costs, "costs")

        if (si + 1) % 10 == 0:
            print(f"  [{phase_name}] {si+1}/{len(samples)} done", flush=True)

    return all_results, all_metrics, all_costs


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=list(PHASES.keys()))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    phase = args.phase
    n_target = PHASES[phase]
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"=== Order Recovery v2: {phase} phase ({n_target} samples) ===")
    print(f"Config: {json.dumps(CONFIG, indent=2)}")

    # Build attack
    print("\nBuilding attack pipeline...")
    atk = build_attack()
    print(f"  Model loaded, {len(atk['all_secs'])}/{atk['H']} heads calibrated")

    # Generate samples
    print(f"\nGenerating {n_target} samples...")

    if phase == "smoke":
        # 3 samples: one with repeats, one multi-block, one edge case
        samples = [
            {"id": "smoke_repeat", "category": "conversation",
             "text": "I think we should grab dinner tonight at that new Italian place."},
            {"id": "smoke_multiblock", "category": "narrative",
             "text": "The young woman had always dreamed of living in Paris. After saving money for three years she finally bought a plane ticket."},
            {"id": "smoke_structured", "category": "structured",
             "text": "My name is Jordan Lee and I can be reached at jordan.lee@mail-example.com or 555-214-8307."},
        ]

    elif phase == "dev":
        # 60 dev samples: 3 categories × 4 length buckets, ~5 per cell
        tok = atk["tok"]
        samples = []
        di = 0
        for cat in ["conversation", "narrative", "structured"]:
            for tb in [1, 2, 4, 8]:
                for j in range(5):
                    text = generate_sample_of_length(cat, tb, tok)
                    samples.append({"id": f"dev_{di:03d}_{cat[:4]}_b{tb}",
                                   "category": cat, "target_blocks": tb, "text": text})
                    di += 1

    elif phase == "main":
        # Distinct seed so dev samples can never reappear in the main test set
        rng.seed(CONFIG["seed_sampling"] + 1)
        # 600 samples: 3 categories × 4 length buckets × 50 each
        tok = atk["tok"]
        samples = []
        idx = 0
        for cat in ["conversation", "narrative", "structured"]:
            for tb in [1, 2, 4, 8]:
                for j in range(50):
                    text = generate_sample_of_length(cat, tb, tok)
                    samples.append({"id": f"main_{idx:04d}_{cat[:4]}_b{tb}",
                                   "category": cat, "target_blocks": tb, "text": text,
                                   "source_family": f"{cat}_b{tb}"})
                    idx += 1

    elif phase == "boundary":
        rng.seed(CONFIG["seed_sampling"] + 2)  # distinct stream from dev/main
        samples = []
        # Boundary A: 100 pairs (same multiset, different fact)
        for i in range(100):
            ta, tb = generate_boundary_pair()
            samples.append({"id": f"ba_{i:03d}_a", "category": "boundary_pair",
                           "text": ta, "pair_id": i, "direction": "a"})
            samples.append({"id": f"ba_{i:03d}_b", "category": "boundary_pair",
                           "text": tb, "pair_id": i, "direction": "b"})
        # Boundary B: 100 random sequences
        for i in range(100):
            ids = generate_random_sequence()
            samples.append({"id": f"rb_{i:03d}", "category": "random_seq",
                           "text": ids})  # already token IDs

    # Save samples manifest (texts + grouping; truth isolated from scorer here)
    samples_path = os.path.join(OUTDIR, f"samples_{phase}.jsonl")
    with open(samples_path, "w") as f:
        for s in samples:
            rec = {k: v for k, v in s.items() if k != "text"}
            if "text" in s:
                rec["text_or_ids"] = s["text"] if isinstance(s["text"], str) else list(s["text"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} sample manifests to {samples_path}")

    # Run experiment with incremental crash-safe writes (each sample flushed on
    # completion; a mid-run failure never loses more than one sample)
    print(f"\nRunning {len(samples)} samples with conditions: "
          f"['end2end', 'random_order', 'oracle_ms', 'no_cross_block']")

    conditions = ["end2end", "random_order", "oracle_ms", "no_cross_block"]
    resume = getattr(args, "resume", False)
    done_ids = set()
    if resume:
        import json as _json
        mpath = os.path.join(OUTDIR, f"metrics_{phase}.jsonl")
        if os.path.exists(mpath):
            with open(mpath) as f:
                for line in f:
                    try:
                        done_ids.add(_json.loads(line)["id"])
                    except Exception:
                        pass
        print(f"  resume: {len(done_ids)} completed samples will be skipped", flush=True)
        with open(os.path.join(OUTDIR, f"resume_info_{phase}.json"), "w") as _f:
            json.dump({"resumed_from": len(done_ids), "ts": time.strftime("%F %T")}, _f)
    sinks = {}
    for name in ("predictions", "metrics", "costs"):
        sinks[name] = open(os.path.join(OUTDIR, f"{name}_{phase}.jsonl"), "a" if resume else "w")
    results, metrics, costs = run_experiment(atk, samples, phase, conditions,
                                             sinks=sinks, skip_ids=done_ids)
    for f in sinks.values():
        f.close()
    print(f"  Artifacts (incrementally written) in {OUTDIR}/")

    # Aggregate summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({phase})")
    print(f"{'='*60}")

    # Aggregate by condition
    for cond in conditions:
        vals = [m[cond]["position_accuracy"] for m in metrics
                if cond in m and "position_accuracy" in m.get(cond, {})]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {cond:15s}: PA = {avg:.1%} (n={len(vals)})")

    # By category
    for cat in ["conversation", "narrative", "structured", "boundary_pair", "random_seq"]:
        vals = [m["end2end"]["position_accuracy"] for m in metrics
                if m.get("category") == cat and "end2end" in m]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  [{cat:15s}] end2end PA = {avg:.1%} (n={len(vals)})")

    print(f"\nDone. Artifacts in {OUTDIR}/")


if __name__ == "__main__":
    main()
