import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bleu import corpus_bleu, ngrams, tokenize


def find(name):
    for p in (HERE / name, HERE / "data" / name, HERE / "dataset" / name):
        if p.exists():
            return p
    raise FileNotFoundError(name)


refs = pd.read_csv(find("references.csv"))
panel = pd.read_csv(find("panel.csv"))
ref_map = refs.groupby(["round", "card"]).reference.apply(list).to_dict()

ENTITIES = sorted([
    "Raja Indian Cuisine", "Rice Boat", "Wrestlers", "Portland Arms", "Cricketers", "Wildwood",
    "Café Rouge", "Ranch", "Mill", "Golden Curry", "All Bar One", "Zizzi", "Café Brazil", "Bakers",
    "Vaults", "Blue Spice", "Loch Fyne", "Punter", "Aromi", "Cocum", "Alimentum", "Café Sicilia",
    "Crowne Plaza", "Rainbow Vegetarian", "Fitzbillies", "Cotto", "Yippee Noodle", "Waterman",
    "Express by Holiday", "Clowns", "Clare Hall", "Strada", "Giraffe", "Phoenix", "Café Adriatic",
    "Browns Cambridge", "Cambridge Blue", "Green Man", "Plough", "Sorrento", "Olive Grove",
    "Twenty Two", "Midsummer House", "Avalon", "Golden Palace", "Dumpling Tree", "Eagle",
    "Bibimbap", "Burger King", "Travellers Rest", "Taste of Cambridge",
], key=len, reverse=True)

FACTS = {
    "food": r"japanese|sushi|french|english|british|italian|indian|chinese|fast[- ]?food|pasta|spaghetti|curry|burger|wine|cheese|breakfast|cuisine|food|fare|dishes|meals?|eats",
    "price": r"pric|cheap|expensive|£|pound|cost|afford|inexpensive|budget|pricey|moderate|high[- ]end|low[- ]cost|mid[- ]?range|\bspend|\bpay|value|dear|bargain|\bunder\b|\bover\b|charge",
    "rating": r"\brat(e|ed|es|ing|ings)\b|star|out of 5|review|customer|acclaim|popular|recommend|\bpoor|excellent|well liked|\bscore",
    "area": r"riverside|river|city cent|centre|center|downtown|waterfront",
    "family": r"famil|kid|child|adult|toddler",
}
FACTS = {k: re.compile(v, re.I) for k, v in FACTS.items()}
FUNCTION_WORDS = set(
    "a an the is are of and with in on at to for near that which it its has have by from or but as "
    "this be there called named serves offers".split()
)
CHECKS = ["omits_facts", "truncated", "broken_grammar", "stutter", "pasted_reference"]


def entities(text):
    s, found = text.lower(), Counter()
    for e in ENTITIES:
        c = s.count(e.lower())
        if c:
            found[e] = c
            s = s.replace(e.lower(), " ")
    return found


def facts(text):
    return {k for k, rx in FACTS.items() if rx.search(text)}


def corpus_stats(texts):
    uni, bi, final = Counter(), Counter(), Counter()
    for x in texts:
        t = tokenize(x)
        uni.update(t)
        bi.update(zip(t, t[1:]))
        for i, w in enumerate(t):
            if i == len(t) - 1 or t[i + 1] in ".!?":
                final[w] += 1
    return uni, bi, final


def ref_coverage(out_tokens, ref_tokens):
    a, b = ngrams(out_tokens, 4), ngrams(ref_tokens, 4)
    if not a or not b:
        return 0.0
    return sum(min(c, b[g]) for g, c in a.items()) / sum(b.values())


def stutter(tokens):
    for k in (1, 2, 3):
        for i in range(len(tokens) - 2 * k + 1):
            span = tokens[i:i + k]
            if span == tokens[i + k:i + 2 * k] and all(w.isalnum() for w in span):
                return True
    return False


UNI, BI, FINAL = corpus_stats(refs.reference)
card_entities, card_facts = {}, {}
for key, texts in ref_map.items():
    ec = Counter(e for x in texts for e in entities(x))
    fc = Counter(f for x in texts for f in facts(x))
    card_entities[key] = {e for e, v in ec.items() if v >= 3}
    card_facts[key] = {f for f, v in fc.items() if v >= 3}


def listener_check(text, key):
    o = tokenize(text)
    ends = [o[i] for i in range(len(o)) if o[i].isalnum() and (i == len(o) - 1 or o[i + 1] in ".!?")]
    return {
        "omits_facts": bool(card_entities[key] - set(entities(text)) or card_facts[key] - facts(text)),
        "truncated": any(UNI[w] >= 30 and FINAL[w] == 0 for w in ends),
        "broken_grammar": any(a in FUNCTION_WORDS and b in FUNCTION_WORDS and BI[(a, b)] == 0 for a, b in zip(o, o[1:])),
        "stutter": stutter(o),
        "pasted_reference": any(len(tokenize(r)) >= 8 and ref_coverage(o, tokenize(r)) >= 0.9 for r in ref_map[key]),
    }


def decode(seats):
    word = ""
    for i in range(0, len(seats), 3):
        d1, d2, d3 = (s - 1 for s in seats[i:i + 3])
        word += chr(ord("A") + 9 * d1 + 3 * d2 + d3)
    return word


def main():
    cards = []
    for t in panel.itertuples():
        row = listener_check(t.text, (t.round, t.card))
        row.update(round=t.round, card=t.card, seat=t.seat, text=t.text)
        cards.append(row)
    cards = pd.DataFrame(cards)
    cards["bluffed"] = cards[CHECKS].any(axis=1)

    decks = []
    for (rnd, seat), deck in panel.groupby(["round", "seat"]):
        b = corpus_bleu(list(deck.text), [ref_map[(rnd, c)] for c in deck.card])
        decks.append({"round": rnd, "seat": seat, "bleu": b["bleu"]})
    decks = pd.DataFrame(decks)
    counts = cards.groupby(["round", "seat"])[CHECKS + ["bluffed"]].sum().reset_index()
    decks = decks.merge(counts, on=["round", "seat"])

    honest = decks.loc[decks.groupby("round").bluffed.idxmin()].sort_values("round")
    bleu_pick = decks.loc[decks.groupby("round").bleu.idxmax()].sort_values("round")
    margins = [sorted(g.bluffed)[1] - sorted(g.bluffed)[0] for _, g in decks.groupby("round")]

    pd.set_option("display.width", 200)
    print("Cards failing the listener check, per deck (of 40):")
    print(decks.pivot(index="round", columns="seat", values="bluffed").to_string())
    print("\nPer check (rows: seat 1, 2, 3; columns: rounds 1..24):")
    for c in CHECKS:
        print(c)
        for row in decks.pivot(index="round", columns="seat", values=c).T.values.tolist():
            print("   ", row)

    seats = [int(s) for s in honest.seat]
    word = decode(seats)
    print("\nHonest seat per round:", seats)
    print("Margin to the next-cleanest deck:", margins)
    print("Honest deck: max flagged cards =", int(honest.bluffed.max()), "| bluffing decks: min flagged cards =",
          int(decks[~decks.index.isin(honest.index)].bluffed.min()))
    strong = decks.assign(s=decks.omits_facts + decks.pasted_reference)
    strong_seats = [int(s) for s in strong.loc[strong.groupby("round").s.idxmin()].sort_values("round").seat]
    strong_margins = [sorted(g.s)[1] - sorted(g.s)[0] for _, g in strong.groupby("round")]
    print("Robustness, only omits_facts + pasted_reference:", strong_seats, "->", decode(strong_seats),
          "| min margin", min(strong_margins))
    print("Rounds where BLEU crowned a bluffer:", int((bleu_pick.seat.values != honest.seat.values).sum()), "of 24")
    print("BLEU's word:   SIGMOID_" + decode([int(s) for s in bleu_pick.seat]))
    print("FLAG:          SIGMOID_" + word)

    out = decks.copy()
    out["honest"] = [int(s == seats[r - 1]) for r, s in zip(out["round"], out.seat)]
    out.round(4).to_csv(HERE / "deck_scores.csv", index=False)
    cards[cards.bluffed][["round", "card", "seat"] + CHECKS + ["text"]].to_csv(HERE / "flagged_cards.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {1: "#2a6fdb", 2: "#e07b39", 3: "#3a9e5f"}
    for seat, g in decks.groupby("seat"):
        axes[0].scatter(g["round"], g.bleu, s=30, color=colors[seat], label=f"seat {seat}")
        axes[1].scatter(g["round"], g.bluffed, s=30, color=colors[seat], label=f"seat {seat}")
    axes[0].scatter(honest["round"], honest.bleu, s=140, facecolors="none", edgecolors="black", label="honest")
    axes[1].scatter(honest["round"], honest.bluffed, s=140, facecolors="none", edgecolors="black", label="honest")
    axes[0].set_ylabel("deck BLEU (house scorer)")
    axes[1].set_ylabel("cards failing listener check")
    axes[1].set_xlabel("round")
    axes[1].set_xticks(range(1, 25))
    axes[0].legend(frameon=False, ncol=4)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"BLEU vs the listener check  ->  SIGMOID_{word}")
    fig.tight_layout()
    fig.savefig(HERE / "listener_check.png", dpi=130)


if __name__ == "__main__":
    main()
