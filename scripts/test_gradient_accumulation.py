ACCUMULATION_STEPS = 4
TOTAL_MICRO_STEPS = 12


def main() -> None:
    micro_step = 0
    optimizer_step = 0
    optimizer_step_microsteps = []

    for _ in range(TOTAL_MICRO_STEPS):
        micro_step += 1

        if micro_step % ACCUMULATION_STEPS == 0:
            optimizer_step += 1
            optimizer_step_microsteps.append(micro_step)

    print("micro steps:", TOTAL_MICRO_STEPS)
    print("optimizer steps:", optimizer_step)
    print(
        "optimizer.step() executed at micro steps:",
        optimizer_step_microsteps,
    )

    assert optimizer_step == 3
    assert optimizer_step_microsteps == [4, 8, 12]

    print("=== GRADIENT ACCUMULATION LOGIC OK ===")


if __name__ == "__main__":
    main()