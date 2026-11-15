"""Find the unregistered White Stork from the two original CSV files.

python main.py --data-dir /path/to/materials --output outputs/findings.json
Only Python's standard library is required. The answer is derived, not embedded.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

STORK = "Ciconia ciconia"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def km_between(observation, registration):
    lat1, lon1 = map(float, (observation["latitude"], observation["longitude"]))
    lat2, lon2 = map(float, (registration["ringed_latitude"], registration["ringed_longitude"]))
    p, q = math.radians(lat1), math.radians(lat2)
    a = (math.sin((q - p) / 2) ** 2 + math.cos(p) * math.cos(q)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1., a)))


def spatial_anchors(observations, register, excluded_species=None):
    """Conservative calibration: unique nearby same-species registered home."""
    by_species = collections.defaultdict(list)
    for entry in register:
        by_species[entry["species"]].append(entry)
    pairs = []
    for row in observations:
        if not row["ring_read"] or row["species"] == excluded_species:
            continue
        year = int(row["date"][:4])
        candidates = sorted(
            (km_between(row, entry), entry["ring"])
            for entry in by_species[row["species"]]
            if int(entry["ringed_year"]) <= year
        )
        if candidates and candidates[0][0] <= 2 and (
            len(candidates) == 1 or candidates[1][0] > 3
        ):
            pairs.append((row, candidates[0][1]))
    return pairs


def learn_classes(pairs, alphabet):
    """Connected components of observed true-letter/read-letter substitutions."""
    parent = {letter: letter for letter in alphabet}

    def find(letter):
        while parent[letter] != letter:
            parent[letter] = parent[parent[letter]]
            letter = parent[letter]
        return letter

    for row, ring in pairs:
        for true, seen in zip(ring, row["ring_read"]):
            if seen != "?":
                parent[find(seen)] = find(true)
    components = collections.defaultdict(list)
    for letter in sorted(alphabet):
        components[find(letter)].append(letter)
    groups = sorted("".join(letters) for letters in components.values())
    lookup = {letter: group for group in groups for letter in group}
    return groups, lookup


def compatible(reading, ring, classes):
    return all(a == "?" or classes[a] == classes[b] for a, b in zip(reading, ring))


def consensus(rows):
    votes = [collections.Counter(row["ring_read"][i] for row in rows
                                 if row["ring_read"][i] != "?") for i in range(6)]
    letters = []
    for position in votes:
        ranking = position.most_common()
        if not ranking or (len(ranking) > 1 and ranking[0][1] == ranking[1][1]):
            raise ValueError("Insufficient evidence: empty or tied position")
        letters.append(ranking[0][0])
    return "".join(letters), votes


def solve(observations, register, excluded_calibration_species=None):
    if len({row["record_id"] for row in observations}) != len(observations):
        raise ValueError("Duplicate record_id")
    if len({row["ring"] for row in register}) != len(register):
        raise ValueError("Duplicate registered ring")
    readings = [row for row in observations if row["ring_read"]]
    for value in [row["ring_read"] for row in readings] + [row["ring"] for row in register]:
        if len(value) != 6 or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ?" for c in value):
            raise ValueError(f"Invalid six-character inscription: {value!r}")
    if any("?" in row["ring"] for row in register):
        raise ValueError("The register must contain complete inscriptions")
    alphabet = set("".join(row["ring"] for row in register))
    alphabet.update("".join(row["ring_read"] for row in readings).replace("?", ""))
    pairs = spatial_anchors(readings, register, excluded_calibration_species)
    groups, classes = learn_classes(pairs, alphabet)
    by_species = collections.defaultdict(list)
    for row in register:
        by_species[row["species"]].append(row["ring"])
    unmatched = []
    candidate_counts = collections.Counter()
    for row in readings:
        matches = [ring for ring in by_species[row["species"]]
                   if compatible(row["ring_read"], ring, classes)]
        candidate_counts[len(matches)] += 1
        if not matches:
            unmatched.append(row)
    locations = {(row["species"], row["locality_id"]) for row in unmatched}
    if len(locations) != 1 or next(iter(locations))[0] != STORK:
        raise ValueError(f"Expected one unexplained White Stork cluster, got {locations}")
    by_checklist = {}
    for row in unmatched:
        old = by_checklist.setdefault(row["checklist_id"], row)
        if old["ring_read"] != row["ring_read"]:
            raise ValueError("Conflicting readings on one checklist")
    cluster = list(by_checklist.values())
    code, votes = consensus(cluster)
    if code in {row["ring"] for row in register}:
        raise ValueError("Recovered inscription is already registered")
    dates = sorted({row["date"] for row in cluster})
    observers = sorted({observer for row in cluster for observer in row["observers"].split("|")})
    date_checks = [consensus([row for row in cluster if row["date"] != date])[0] == code
                   for date in dates]
    observer_checks = [consensus([row for row in cluster
                                 if observer not in row["observers"].split("|")])[0] == code
                       for observer in observers]
    visible = [sum(position.values()) for position in votes]
    winning = [position.most_common(1)[0][1] for position in votes]
    summary = {
        "observation_rows": len(observations), "registered_rings": len(register),
        "ring_readings": len(readings), "calibration_pairs": len(pairs),
        "excluded_calibration_species": excluded_calibration_species,
        "confusion_groups": groups, "registered_candidates_histogram": dict(sorted(candidate_counts.items())),
        "unexplained_readings": len(unmatched), "unexplained_species": STORK,
        "locality_id": cluster[0]["locality_id"], "district": cluster[0]["district"],
        "latitude": cluster[0]["latitude"], "longitude": cluster[0]["longitude"],
        "unique_checklists": len(cluster), "unique_dates": len(dates),
        "unique_observers": len(observers), "date_range": [dates[0], dates[-1]],
        "years": sorted({date[:4] for date in dates}),
        "visible_votes_per_position": visible, "winning_votes_per_position": winning,
        "exact_complete_readings": sum(row["ring_read"] == code for row in cluster),
        "leave_one_date_out": {"unchanged": sum(date_checks), "total": len(date_checks)},
        "leave_one_observer_out": {"unchanged": sum(observer_checks), "total": len(observer_checks)},
        "platform_confirmed": False,
    }
    return {"ring": code, "flag": "SIGMOID_" + code, "summary": summary,
            "position_votes": [dict(position) for position in votes],
            "record_ids": [row["record_id"] for row in cluster]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/findings.json"))
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--exclude-storks-from-calibration", action="store_true")
    args = parser.parse_args()
    observations = read_csv(args.data_dir / "observations.csv")
    register = read_csv(args.data_dir / "ring_register.csv")
    result = solve(observations, register, STORK if args.exclude_storks_from_calibration else None)
    result["summary"]["input_sha256"] = {
        name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
        for name in ["observations.csv", "ring_register.csv"]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False)
                                       + "\n", encoding="utf-8")
    print(result["flag"])
    print(f"Unexplained readings: {result['summary']['unexplained_readings']}")
    print(f"Evidence saved to {args.output}")


if __name__ == "__main__":
    main()
