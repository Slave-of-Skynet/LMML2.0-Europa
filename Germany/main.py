"""Discover a multilinear response plus a planar hard step within 150 queries.

Only solve(oracle, points) is exposed. No practice coefficients, helper files,
grader access, or knowledge of the test function is used by this procedure.
"""
from itertools import combinations, product
import numpy as np


NAMES = ["x1", "x2", "x3", "x4", "x5"]
PAIRS = list(combinations(range(5), 2))


def two_levels(values):
    levels = np.array([min(values), max(values)], dtype=float)
    for _ in range(20):
        labels = np.abs(values[:, None] - levels).argmin(axis=1)
        new = np.array([np.mean(values[labels == k]) if np.any(labels == k) else levels[k]
                        for k in range(2)])
        if np.max(np.abs(new - levels)) < 1e-10:
            break
        levels = new
    return labels, levels


def design(X, pair, gate):
    return np.column_stack([np.ones(len(X)), X, X[:, pair[0]] * X[:, pair[1]], gate])


def solve(oracle, points):
    cache = {}
    observations = []

    def query(x, phase):
        x = np.clip(np.asarray(x, dtype=float), 0, 1)
        key = tuple(round(float(v), 9) for v in x)
        if key in cache:
            return cache[key]
        if oracle.remaining <= 0:
            return None
        y = float(oracle.ask(x.tolist(), phase=phase))
        cache[key] = y
        observations.append((x.copy(), y))
        return y

    # Joint extreme settings expose switches that every centered OWAT misses.
    corners = np.array(list(product((0.0, 1.0), repeat=5)))
    corner_y = np.array([query(x, "factorial-corners") for x in corners])
    rng = np.random.default_rng(2039)
    stencils = []
    h = 0.035

    def stencil():
        anchor = rng.uniform(0.12, 0.82, 5)
        y0 = query(anchor, "local-base")
        single = []
        for i in range(5):
            p = anchor.copy()
            p[i] += h
            single.append(query(p, "local-first-difference"))
        gradient = (np.array(single) - y0) / h
        mixed = []
        for i, j in PAIRS:
            p = anchor.copy()
            p[i] += h
            p[j] += h
            value = query(p, "local-mixed-difference")
            mixed.append((value - single[i] - single[j] + y0) / h**2)
        stencils.append((anchor, y0, gradient, np.array(mixed)))

    stencil()
    stencil()

    def discover():
        X = np.array([p for p, _ in observations])
        y = np.array([v for _, v in observations])
        best = None
        for pair_number, pair in enumerate(PAIRS):
            estimates = [item[3][pair_number] for item in stencils]
            candidates = estimates + [min(estimates, key=abs)]
            for coefficient in candidates:
                if abs(coefficient) < 1e-5:
                    continue
                for anchor, y0, gradient, _ in stencils:
                    linear = gradient.copy()
                    linear[pair[0]] -= coefficient * anchor[pair[1]]
                    linear[pair[1]] -= coefficient * anchor[pair[0]]
                    intercept = y0 - linear @ anchor - coefficient * anchor[pair[0]] * anchor[pair[1]]
                    residual = y - (intercept + X @ linear + coefficient * X[:, pair[0]] * X[:, pair[1]])
                    gate, levels = two_levels(residual)
                    A = design(X, pair, gate)
                    beta = np.linalg.lstsq(A, y, rcond=None)[0]
                    error = float(np.mean((A @ beta - y)**2))
                    if best is None or error < best[0]:
                        best = (error, pair, beta, gate)
        return best

    best = discover()
    # A third independently placed neighbourhood resolves a stencil crossing
    # the step instead of mistaking its large finite difference for curvature.
    if (best is None or best[0] > 1e-7) and oracle.remaining >= 65:
        stencil()
        best = discover()
    if best is None:
        # Safe fallback retains a valid prediction without risking query 151.
        X = np.array([x for x, _ in observations])
        y = np.array([v for _, v in observations])
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
        return {"relevant_variables": [NAMES[i] for i in range(5) if abs(beta[i + 1]) > 0.001],
                "irrelevant_variables": [NAMES[i] for i in range(5) if abs(beta[i + 1]) <= 0.001],
                "interactions": [], "discontinuities": [],
                "predictions": [float(beta[0] + np.dot(beta[1:], x)) for x in points],
                "notes": "No stable two-regime bilinear model found; conservative linear fallback."}

    error, pair, beta, gate = best
    if beta[-1] < 0:
        beta[0] += beta[-1]
        beta[-1] *= -1

    def smooth(x):
        return beta[0] + np.dot(beta[1:6], x) + beta[6] * x[pair[0]] * x[pair[1]]

    def state(x, phase="switch-classification"):
        value = query(x, phase)
        if value is None:
            return None
        return int(value - smooth(x) >= beta[-1] * 0.5)

    corner_gate = np.array([state(x) for x in corners])
    low = np.flatnonzero(corner_gate == 0)
    high = np.flatnonzero(corner_gate == 1)
    normal = np.zeros(5)
    threshold = 0.0
    gate_variables = []
    if len(low) and len(high):
        # A long diagonal gives an interior boundary point, unlike repeatedly
        # probing a face on which a second gate input may be invisible.
        a, b = max(((i, j) for i in low for j in high),
                   key=lambda ij: np.sum((corners[ij[0]] - corners[ij[1]])**2))
        left, right = corners[a].copy(), corners[b].copy()
        for _ in range(12):
            if oracle.remaining <= 30:
                break
            mid = (left + right) / 2
            if state(mid, "diagonal-boundary-bisection"):
                right = mid
            else:
                left = mid
        boundary = (left + right) / 2
        endpoints = {}
        for i in range(5):
            if oracle.remaining < 2:
                break
            p0, p1 = boundary.copy(), boundary.copy()
            p0[i], p1[i] = 0, 1
            s0, s1 = state(p0, "gate-axis-endpoints"), state(p1, "gate-axis-endpoints")
            if s0 != s1:
                endpoints[i] = (s0, s1)

        if endpoints:
            influence = []
            for i in endpoints:
                count = 0
                for j, x in enumerate(corners):
                    if x[i] == 0:
                        other = x.copy()
                        other[i] = 1
                        k = next(k for k, p in enumerate(corners) if np.array_equal(p, other))
                        count += int(corner_gate[j] != corner_gate[k])
                influence.append((count, i))
            primary = max(influence)[1]
            sign = 1.0 if endpoints[primary][1] else -1.0

            def crossing(fixed, iterations):
                p0, p1 = fixed.copy(), fixed.copy()
                p0[primary], p1[primary] = 0, 1
                s0, s1 = state(p0), state(p1)
                if s0 is None or s1 is None or s0 == s1:
                    return None
                lo, hi = 0.0, 1.0
                for _ in range(iterations):
                    if oracle.remaining <= 0:
                        break
                    mid = (lo + hi) / 2
                    p = fixed.copy()
                    p[primary] = mid
                    if state(p, "axis-boundary-bisection") == s0:
                        lo = mid
                    else:
                        hi = mid
                return (lo + hi) / 2

            count = len(endpoints)
            iterations = min(17, max(5, (oracle.remaining - 2 * count - 8) // count))
            root = crossing(boundary, iterations)
            if root is not None:
                boundary[primary] = root
                normal[primary] = sign
                margin = max(0.0001, min(root, 1 - root))
                for other in endpoints:
                    if other == primary or oracle.remaining < 4:
                        continue
                    delta = min(0.18, 0.35 * margin)
                    delta *= 1 if boundary[other] < 0.5 else -1
                    shifted = None
                    for _ in range(5):
                        if oracle.remaining < 4:
                            break
                        fixed = boundary.copy()
                        fixed[other] = np.clip(fixed[other] + delta, 0, 1)
                        actual_delta = fixed[other] - boundary[other]
                        shifted = crossing(fixed, min(iterations, max(1, oracle.remaining - 2)))
                        if shifted is not None:
                            break
                        # Boolean corner influence can tie even for unequal
                        # plane weights. Shrink the displacement until the
                        # new boundary remains inside the primary axis range.
                        delta *= 0.5
                    if shifted is not None and abs(actual_delta) > 1e-8:
                        normal[other] = -sign * (shifted - root) / actual_delta
                threshold = float(normal @ boundary)
                gate_variables = [NAMES[i] for i in endpoints]

    # Check fresh joint settings, and refine smooth coefficients only away from
    # the estimated surface. Bisection points themselves are intentionally not
    # used as noisy/ambiguous regime labels in the final regression.
    refinement = list(observations[:64])
    if np.linalg.norm(normal) > 0:
        for _ in range(min(8, oracle.remaining)):
            x = rng.uniform(0, 1, 5)
            value = query(x, "joint-validation")
            refinement.append((x, value))
        stable = [(x, y) for x, y in refinement
                  if abs(normal @ x - threshold) / np.linalg.norm(normal) > 0.0025]
        if len(stable) >= 12:
            X = np.array([x for x, _ in stable])
            y = np.array([y for _, y in stable])
            beta = np.linalg.lstsq(design(X, pair, (X @ normal >= threshold).astype(float)), y, rcond=None)[0]

    relevant = set(pair)
    relevant.update(i for i in range(5) if abs(beta[i + 1]) > 1e-5)
    relevant.update(i for i in range(5) if NAMES[i] in gate_variables)
    base = " + ".join([repr(float(beta[0]))] + [f"({float(beta[i + 1])!r})*{NAMES[i]}" for i in range(5)])
    base += f" + ({float(beta[6])!r})*{NAMES[pair[0]]}*{NAMES[pair[1]]}"
    condition = " + ".join(f"({float(normal[i])!r})*{NAMES[i]}" for i in range(5)) + f" >= {threshold!r}"
    expression = base + f" + ({float(beta[-1])!r} if {condition} else 0.0)"
    predictions = [float(smooth(x) + (beta[-1] if normal @ x >= threshold else 0)) for x in points]
    return {
        "relevant_variables": [NAMES[i] for i in sorted(relevant)],
        "irrelevant_variables": [NAMES[i] for i in range(5) if i not in relevant],
        "interactions": [{"variables": [NAMES[i] for i in pair], "type": "product", "coefficient": float(beta[6])}],
        "discontinuities": [{"gate_variables": gate_variables,
                             "boundary": {"form": "linear", "coefficients": dict(zip(NAMES, normal.tolist())),
                                          "threshold": threshold, "active_when": "greater_equal"},
                             "jump": float(beta[-1]), "condition": condition}] if gate_variables else [],
        "model_expression": expression,
        "predictions": predictions,
        "notes": f"Joint factorial probes, independent mixed finite differences, two-regime fit, and adaptive boundary bisection. Discovery MSE={error:.3g}; {oracle.spent} charged queries. Query design is independent of challenge points.",
    }
