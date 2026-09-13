import numpy as np
import itertools
import math
import time
import csv
import os

# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "jsn-vectors.csv"

TOP_RESULTS = 10

# Enter specific combinations of TypeIDs to look up in results.
# Example:
# SPECIFIC_COMBINATIONS = [
#     (77825, 77826, 77830, 77835),
#     (77840, 77841, 78582, 78586),
# ]
SPECIFIC_COMBINATIONS = [
(77826, 77837, 77844, 78586),
(77837, 78583, 78584, 78586),
(77829, 77835, 77839, 77845),
(77837, 77844, 78582, 78584),
]

NUM_TARGETS = 300
MAX_MOVES = 5
RADIUS_LIMIT = 1496000000000
ANGLE_LIMIT_DEGREES = 20.0

RESULTS_FILE = "jsn-coverage-results.csv"


# ============================================================
# LOAD VECTOR CSV
# ============================================================

data = np.genfromtxt(
    CSV_FILE,
    delimiter=",",
    names=True
)

type_ids = data["TypeID"].astype(int)

V = np.column_stack((
    data["X"],
    data["Y"],
    data["Z"]
)).astype(float)

NUM_VECTORS = len(V)


# ============================================================
# VALIDATE INPUT
# ============================================================

if V.ndim != 2 or V.shape[1] != 3:
    raise ValueError(
        "The vector data must contain exactly three coordinate columns: X, Y, Z."
    )

if NUM_VECTORS < 4:
    raise ValueError(
        "At least four vectors are required."
    )

print(f"Loaded {NUM_VECTORS} vectors.")


# ============================================================
# GENERATE TARGET DIRECTIONS
# ============================================================

i = np.arange(NUM_TARGETS)

z = 1.0 - 2.0 * (i + 0.5) / NUM_TARGETS
r = np.sqrt(1.0 - z * z)

golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
theta = 2.0 * np.pi * i / golden_ratio

D = np.column_stack((
    r * np.cos(theta),
    r * np.sin(theta),
    z
))

ANGLE_THRESHOLD = math.cos(
    math.radians(ANGLE_LIMIT_DEGREES)
)


# ============================================================
# GENERATE ALL POSSIBLE MOVE SEQUENCES
# ============================================================

def generate_sequences(length):

    count = 4 ** length

    sequences = np.empty(
        (count, length),
        dtype=np.int8
    )

    numbers = np.arange(count)

    for column in range(length - 1, -1, -1):
        sequences[:, column] = numbers % 4
        numbers //= 4

    return sequences


SEQUENCES = {
    length: generate_sequences(length)
    for length in range(1, MAX_MOVES + 1)
}


# ============================================================
# CONVERT SEQUENCES TO VECTOR-USE COUNTS
# ============================================================

def sequence_coefficients(sequences):

    coefficients = np.zeros(
        (len(sequences), 4),
        dtype=np.int8
    )

    for vector_index in range(4):
        coefficients[:, vector_index] = np.sum(
            sequences == vector_index,
            axis=1
        )

    return coefficients


COEFFICIENTS = {
    length: sequence_coefficients(SEQUENCES[length])
    for length in range(1, MAX_MOVES + 1)
}


# ============================================================
# PRECOMPUTE PREFIX COEFFICIENTS
# ============================================================

# Navigators stop working when distance exceeds RADIUS_LIMIT.
# Intermediate positions must remain within RADIUS_LIMIT.
# The final position is allowed to exceed RADIUS_LIMIT.

PREFIX_COEFFICIENTS = {}

for length in range(2, MAX_MOVES + 1):
    prefixes = []
    sequences = SEQUENCES[length]

    for prefix_length in range(1, length):
        prefix = sequences[:, :prefix_length]

        prefix_coeffs = np.zeros(
            (len(sequences), 4),
            dtype=np.int8
        )

        for vector_index in range(4):
            prefix_coeffs[:, vector_index] = np.sum(
                prefix == vector_index,
                axis=1
            )

        prefixes.append(prefix_coeffs)

    PREFIX_COEFFICIENTS[length] = prefixes


# ============================================================
# COVERAGE CALCULATION
# ============================================================

def coverage_for(vecs):

    all_endpoints = []

    for length in range(1, MAX_MOVES + 1):
        coeffs = COEFFICIENTS[length]

        endpoints = coeffs @ vecs

        if length > 1:
            valid = np.ones(
                len(endpoints),
                dtype=bool
            )

            for prefix_coeffs in PREFIX_COEFFICIENTS[length]:
                prefix_positions = prefix_coeffs @ vecs

                distance_squared = np.sum(
                    prefix_positions * prefix_positions,
                    axis=1
                )

                valid &= (
                    distance_squared <= RADIUS_LIMIT * RADIUS_LIMIT
                )

                if not np.any(valid):
                    break

            endpoints = endpoints[valid]

        if len(endpoints):
            all_endpoints.append(endpoints)

    if not all_endpoints:
        return 0.0, 0

    ends = np.vstack(all_endpoints)

    # Remove zero-length endpoints because they have no direction.
    norms_squared = np.sum(
        ends * ends,
        axis=1
    )

    valid = norms_squared > 1e-24
    ends = ends[valid]
    norms = np.sqrt(norms_squared[valid])

    if len(ends) == 0:
        return 0.0, 0

    U = ends / norms[:, None]

    max_dot = np.max(
        U @ D.T,
        axis=0
    )

    covered = max_dot >= ANGLE_THRESHOLD
    coverage = np.mean(covered)

    return coverage, len(ends)


# ============================================================
# LOAD EXISTING RESULTS OR RUN THE FULL SIMULATION
# ============================================================

def load_results_from_csv(filename):

    loaded_results = []

    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "Rank",
            "TypeID_1",
            "TypeID_2",
            "TypeID_3",
            "TypeID_4",
            "Coverage_Percent",
            "Reachable_Endpoints",
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{filename} does not contain the expected result columns."
            )

        for row in reader:
            type_id_combination = (
                int(row["TypeID_1"]),
                int(row["TypeID_2"]),
                int(row["TypeID_3"]),
                int(row["TypeID_4"]),
            )

            coverage = (
                float(row["Coverage_Percent"]) / 100.0
            )

            endpoint_count = int(
                row["Reachable_Endpoints"]
            )

            loaded_results.append((
                coverage,
                type_id_combination,
                endpoint_count
            ))

    # Sort by coverage in case the file was manually edited.
    loaded_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return loaded_results


total_combinations = math.comb(
    NUM_VECTORS,
    4
)

overall_start_time = time.perf_counter()

if os.path.exists(RESULTS_FILE):
    print(
        f"\nResults file found: {RESULTS_FILE}"
    )
    print(
        "Loading existing results...\n"
    )

    results = load_results_from_csv(
        RESULTS_FILE
    )

    expected_results = total_combinations

    if len(results) != expected_results:
        print(
            f"WARNING: Expected {expected_results} combinations, "
            f"but found {len(results)} in the existing file."
        )

    elapsed_time = (
        time.perf_counter() - overall_start_time
    )

    print(
        f"Loaded {len(results)} results in "
        f"{elapsed_time:.2f} seconds."
    )

else:
    print(
        "\nNo results file found."
    )
    print(
        "Running simulation...\n"
    )

    print(f"Vectors:       {NUM_VECTORS}")
    print(f"Combinations:  {total_combinations:,}")
    print(f"Target points: {NUM_TARGETS:,}")
    print(f"Max moves:     {MAX_MOVES}")
    print(f"Angle limit:   {ANGLE_LIMIT_DEGREES} degrees")
    print(f"Radius limit:  {RADIUS_LIMIT:,}")
    print()

    print("-" * 70)


    results = []

    simulation_start_time = time.perf_counter()

    for k, combination in enumerate(
        itertools.combinations(
            range(NUM_VECTORS),
            4
        ),
        start=1
    ):
        vecs = V[list(combination)]

        coverage, endpoint_count = coverage_for(
            vecs
        )

        type_id_combination = tuple(
            int(type_ids[index])
            for index in combination
        )

        results.append((
            coverage,
            type_id_combination,
            endpoint_count
        ))

        if k % 5000 == 0 or k == total_combinations:
            elapsed = (
                time.perf_counter() - simulation_start_time
            )

            rate = k / elapsed

            remaining = (
                (total_combinations - k) / rate
                if rate > 0
                else 0
            )

            print(
                f"{k:>7,} / {total_combinations:,} "
                f"({100*k/total_combinations:6.2f}%) | "
                f"{elapsed:8.1f}s elapsed | "
                f"{remaining:8.1f}s remaining"
            )

    print("-" * 70)
    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Rank",
            "TypeID_1",
            "TypeID_2",
            "TypeID_3",
            "TypeID_4",
            "Coverage_Percent",
            "Reachable_Endpoints",
        ])

        for rank, (
            coverage,
            type_id_combination,
            endpoint_count
        ) in enumerate(
            results,
            start=1
        ):
            writer.writerow([
                rank,
                type_id_combination[0],
                type_id_combination[1],
                type_id_combination[2],
                type_id_combination[3],
                coverage * 100.0,
                endpoint_count,
            ])

    elapsed_time = (
        time.perf_counter() - overall_start_time
    )

    print(
        f"\nResults saved to {RESULTS_FILE}"
    )
    print(
        f"Simulation completed in {elapsed_time:.2f} seconds."
    )


# ============================================================
# DISPLAY TOP RESULTS
# ============================================================

print(f"\nTop {TOP_RESULTS} combinations:\n")

print(
    f"{'Rank':>4}  "
    f"{'TypeIDs':<28} "
    f"{'Coverage':>10} "
    f"{'Endpoints':>10}"
)

print("-" * 60)

for rank, (
    coverage,
    type_id_combination,
    endpoint_count
) in enumerate(
    results[:TOP_RESULTS],
    start=1
):
    print(
        f"{rank:>4}. "
        f"{type_id_combination} "
        f"{coverage * 100:>9.2f}% "
        f"{endpoint_count:>10}"
    )


# ============================================================
# SPECIFIC COMBINATION LOOKUP
# ============================================================

if SPECIFIC_COMBINATIONS:
    print(
        "\n\nSpecific combination lookup:\n"
    )

    print(
        f"{'Rank':>4}  "
        f"{'TypeIDs':<28} "
        f"{'Coverage':>10} "
        f"{'Endpoints':>10}"
    )

    print("-" * 60)


    # Build a lookup dictionary using the TypeID.
    results_by_combination = {
        tuple(sorted(type_id_combination)): (
            coverage,
            endpoint_count,
            rank
        )
        for rank, (
            coverage,
            type_id_combination,
            endpoint_count
        ) in enumerate(
            results,
            start=1
        )
    }

    for requested_combination in SPECIFIC_COMBINATIONS:
        normalized_combination = tuple(
            sorted(
                int(type_id)
                for type_id in requested_combination
            )
        )

        if len(normalized_combination) != 4:
            print(
                f"TypeIDs {requested_combination}: "
                "ERROR — exactly four TypeIDs are required."
            )
            continue

        if len(set(normalized_combination)) != 4:
            print(
                f"TypeIDs {requested_combination}: "
                "ERROR — TypeIDs must be unique."
            )
            continue

        result = results_by_combination.get(
            normalized_combination
        )

        if result is None:
            print(
                f"TypeIDs {requested_combination}: "
                "combination not found."
            )
            continue

        coverage, endpoint_count, rank = result

        print(
            f"{rank:>4}. "
            f"{normalized_combination} "
            f"{coverage * 100:>9.2f}% "
            f"{endpoint_count:>10}"
        )



# ============================================================
# FINAL ELAPSED TIME
# ============================================================

total_elapsed_time = (
    time.perf_counter() - overall_start_time
)

print(
    f"\nTotal elapsed time: {total_elapsed_time:.2f} seconds."
)