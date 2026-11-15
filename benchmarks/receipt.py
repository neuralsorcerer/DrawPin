"""Independent replay of physical buffer events from sampler receipts."""

from collections import Counter, defaultdict
from itertools import zip_longest


def load_receipt(blob):
    # Futures are decoded from checked P records (never from the native pool).
    lanes = defaultdict(list)
    for line in blob.splitlines():
        f = line.split(b",")
        lanes[int(f[1])].append(f)
    result = {}
    for i, records in lanes.items():
        stream = []
        later = None
        for f in records[::-1]:
            if f[0] == b"K":
                stream.append(("tick", None))
                if f[-1] == b"2":
                    assert later is not None and later[0] == 1, (i, f, later)
                    stream.append(("hold", later[1]))
                else:
                    assert f[-1] == b"0"
            elif f[0] == b"T":
                stream.append(("drop", None))
            elif f[0] == b"P":
                later = int(f[6]), int(f[5])
                assert later[0] in (0, 1)
                if later[0]:
                    stream.append(("read", later[1]))
            else:
                raise ValueError(f[0])
        ticks = []
        queue = []
        for k, v in reversed(stream):
            if k == "tick":
                ticks.append(queue)
                queue = []
            else:
                queue.append((k, v))
        assert not queue
        result[i] = ticks
    return result


def simulate(waves, width, mode, isolate=False):
    stats = Counter()
    for group in waves:
        # For lease policies store only distinct keys: values are sets of
        # owning streams. Dictionaries list keys from least to most recent.
        slots = {}
        lane_key = {}
        private = {}
        lru = {}
        waiting = {}
        ctrl = int(mode[3:]) if mode.startswith("lru") else 0
        stats["page_bytes"] = max(stats["page_bytes"], 4096 * (len(group) + ctrl))

        def drop(lane):
            if lane in lane_key:
                old = lane_key.pop(lane)
                owners = slots[old]
                owners.remove(lane)
                if not owners and mode == "alias":
                    del slots[old]

        def update():
            if mode in ("lease", "ticket", "alias"):
                n = len(slots)
            else:
                n = len(private) + len(lru)
            stats["peak_valid_pages"] = max(stats["peak_valid_pages"], n)

        for page_time in zip_longest(*(events for events in group.values()), fillvalue=()):
            for lane, events in zip(group, page_time):
                for event, sid in events:
                    if event == "drop":
                        drop(lane)
                        continue
                    if event == "hold":
                        if lane in waiting:
                            assert waiting[lane] == sid
                        waiting[lane] = sid
                        continue
                    assert event == "read"
                    stats["logical"] += 1
                    key = (lane if isolate else 0, sid)
                    promised = lane in waiting
                    if promised:
                        assert waiting.pop(lane) == sid
                        if mode == "ticket":
                            stats["announced_reads"] += 1
                    if mode == "private":
                        stats["backing"] += 1
                        private[lane] = key
                    elif ctrl:
                        if key in lru:
                            del lru[key]
                            stats["hit_lru"] += 1
                        else:
                            if len(lru) == ctrl:
                                del lru[next(iter(lru))]
                            stats["backing"] += 1
                        lru[key] = True
                        private[lane] = key
                        stats["copied_bytes"] += 4096
                    elif mode == "last":
                        holders = [i for i, v in private.items() if v == key]
                        if holders:
                            stats["hit_last"] += 1
                            if min(holders) != lane:
                                stats["copied_bytes"] += 4096
                        else:
                            stats["backing"] += 1
                        private[lane] = key
                    else:
                        drop(lane)
                        if key in slots:
                            owners = slots.pop(key)
                            stats["hit_pinned" if owners else "hit_idle"] += 1
                            if mode == "ticket" and promised:
                                stats["announced_hits"] += 1
                        else:
                            if len(slots) == len(group):
                                free = [k for k, owners in slots.items() if not owners]
                                assert free  # old logical buffer already unpinned
                                if mode == "ticket":
                                    targets = Counter(
                                        (i if isolate else 0, s) for i, s in waiting.items()
                                    )
                                    victim = min(free, key=targets.__getitem__)
                                    stats["overrides"] += victim != free[0]
                                else:
                                    victim = free[0]
                                del slots[victim]
                            owners = set()
                            stats["backing"] += 1
                        owners.add(lane)
                        slots[key] = owners
                        lane_key[lane] = key
                        stats["max_pins_on_one_page"] = max(
                            stats["max_pins_on_one_page"], len(owners)
                        )
                    update()
        assert not lane_key and not waiting
    if mode in ("private", "last") or mode.startswith("lru"):
        stats["max_pins_on_one_page"] = int(stats["logical"] != 0)
    return stats


def groups(streams, width):
    names = sorted(streams)
    return [
        {i: streams[i] for i in names[start : start + width]}
        for start in range(0, len(names), width)
    ]
