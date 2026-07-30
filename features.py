import math
import statistics

# GripAngle intentionally excluded: no honest browser-equivalent signal
# exists (stylus tilt/twist support is inconsistent across devices and
# browsers, and a plain mouse reports none at all).
AGG_FUNCS = {
    "Velocity": ["mean", "std", "max"],
    "Acceleration": ["mean", "std", "max"],
    "Jerk": ["mean", "std", "max"],
    "Pressure": ["mean", "std"],
    "Pressure_RollStd": ["mean", "max"],
}

# Jerk needs 3 successive diffs and the rolling pressure-std needs a
# handful of samples to be meaningful - below this many surviving rows,
# aggregated stats are noise, not signal.
MIN_KINEMATIC_ROWS = 10


def _rolling_std(values: list[float], window: int) -> float | None:
    if len(values) < 2:
        return None
    last_values = values[-window:]
    if len(last_values) < 2:
        return None
    try:
        return statistics.stdev(last_values)
    except statistics.StatisticsError:
        return None


def engineer_kinematics(group: list[dict]) -> list[dict]:
    group = sorted(group, key=lambda row: row["Timestamp"])
    count = len(group)
    if count < 4:
        return []

    timestamps = [row["Timestamp"] for row in group]
    xs = [row["X"] for row in group]
    ys = [row["Y"] for row in group]
    pressures = [row["Pressure"] for row in group]

    dt = [None] * count
    velocity = [None] * count
    acceleration = [None] * count
    jerk = [None] * count

    for i in range(1, count):
        delta_t = timestamps[i] - timestamps[i - 1]
        if delta_t <= 0:
            continue

        dt[i] = delta_t
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        distance = math.sqrt(dx * dx + dy * dy)
        velocity[i] = distance / delta_t

        if velocity[i - 1] is not None and dt[i - 1] and dt[i - 1] > 0:
            acceleration[i] = (velocity[i] - velocity[i - 1]) / delta_t

        if acceleration[i] is not None and acceleration[i - 1] is not None:
            jerk[i] = (acceleration[i] - acceleration[i - 1]) / delta_t

    positive_dts = [value for value in dt[1:] if value is not None and value > 0]
    median_dt = statistics.median(positive_dts) if positive_dts else None
    window = max(3, int(round(50 / median_dt))) if median_dt and median_dt > 0 else 5

    results = []
    for i in range(count):
        if velocity[i] is None or acceleration[i] is None or jerk[i] is None:
            continue

        pressure_rollstd = _rolling_std(pressures[: i + 1], window)
        if pressure_rollstd is None:
            continue

        results.append(
            {
                "X": xs[i],
                "Y": ys[i],
                "Timestamp": timestamps[i],
                "Pressure": pressures[i],
                "Velocity": velocity[i],
                "Acceleration": acceleration[i],
                "Jerk": jerk[i],
                "Pressure_RollStd": pressure_rollstd,
            }
        )

    return results


def compute_point_kinematics(rows: list[dict], group_cols: list) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        groups.setdefault(key, []).append(row)

    results = []
    for group in groups.values():
        results.extend(engineer_kinematics(group))

    return results


def aggregate_feature_table(rows: list[dict], group_cols: list, agg_funcs: dict = AGG_FUNCS) -> list[dict]:
    if not rows:
        return []

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        groups.setdefault(key, []).append(row)

    table = []
    for key, group_rows in groups.items():
        aggregated = {}
        for index, group_col in enumerate(group_cols):
            aggregated[group_col] = key[index]

        for base, stats in agg_funcs.items():
            values = [entry[base] for entry in group_rows]
            for stat in stats:
                column_name = f"{base}_{stat}"
                if stat == "mean":
                    aggregated[column_name] = statistics.mean(values) if values else float("nan")
                elif stat == "std":
                    aggregated[column_name] = statistics.stdev(values) if len(values) >= 2 else float("nan")
                elif stat == "max":
                    aggregated[column_name] = max(values) if values else float("nan")
                else:
                    raise ValueError(f"Unsupported agg stat: {stat}")

        table.append(aggregated)

    return table


def feature_column_names(agg_funcs: dict = AGG_FUNCS) -> list[str]:
    return [f"{base}_{stat}" for base, stats in agg_funcs.items() for stat in stats]
