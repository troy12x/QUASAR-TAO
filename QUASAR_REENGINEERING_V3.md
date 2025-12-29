# Quasar Re-Engineering V3: The Semantic Dependency Ridge

## 1. The Vulnerability: Why V2 Failed
V2 introduced "Indirection" (Prompt -> Entity -> Code), but it left two critical holes:
1.  **Brute Force Execution**: A miner executes all 50 functions. They get 50 numbers. If the expected format is "a number," they can just guess or check against a simple heuristic, or send all of them.
2.  **Self-Contained Logic**: The function `def calc(x): return x * 0.88` contains all the logic needed to run it. The miner can ignore the narrative that *explains* the 0.88.

## 2. The Solution: Semantic Dependencies

We must ensure that the code itself is **incomplete** without the narrative state.

### Core Concept: "The Incomplete Function"
The function will no longer be: `def solve(x): return x * 0.88`
It will be:
```python
def solve(x, system_status):
    if system_status == "CRITICAL":
        return (x * 0.88) / 2
    else:
        return x * 0.88
```
**Mechanism**:
- The variable `system_status` is NOT PROVIED in the function call arguments blindly.
- The Miner must **Read the Narrative** to find out: "Is the Gorgon Drive in CRITICAL state or NORMAL state?"
- If they guess wrong (Brute Force), they have a 50/50 chance of failure per function.
- If we chain 3 such dependencies, the chance of guessing drops to 12.5%.

### Core Concept: "The Narrative Invariant" (Breaking Brute Force Selection)
Even if they run all functions, they get 50 results. How do we stop them from knowing which one is right?
**Constraint**: "The correct system output must match the **Sector 7 Parity Rule**."
- **Narrative**: "...Sector 7 protocols require all valid transmissions to be **Odd** numbers..."
- **Design**:
    - Target Function (Gorgon): Returns `441` (Odd) -> **VALID**.
    - Distractor A (Chimera): Returns `440` (Even) -> **INVALID**.
    - Distractor B (Hydra): Returns `442` (Even) -> **INVALID**.
- **Result**: The Miner executes all 50. They see 10 Odd results and 40 Even results.
- They *still* don't know which of the 10 Odd results is right unless they know the prompt asked for "Gorgon Drive".

## 3. Implementation Plan (V3)

### Phase 1: `ContextualNeedleLoader` Upgrade
- **State Injection**: Generate a "State" (e.g. `status="OPTIMIZED"`).
- **Conditional Logic**: Generate functions that use `if/else` based on that state.
- **Invariant Generation**: Ensure the **Target** satisfies a narrative rule (e.g. Divisibility, Parity, Range) and Distractors do not.

### Phase 2: Protocol Support (`challenge_input` enhancement)
- The validator sends `challenge_input = {'x': 100}`.
- The miner must find `system_status` from text to run `solve(100, system_status)`.

### Phase 3: Validation Logic
- Validator runs the logic with the *correct* extracted state.
- Validator checks if the result satisfies the Invariant.

## 4. Security Verification

| Attack Vector | V2 Vulnerability | V3 Defense |
| :--- | :--- | :--- |
| **Brute Force Execution** | Executing all gives 50 valid numbers. | Executing all gives 50 numbers, but Miner doesn't know the inputs (System Status) so results are garbage. |
| **Symbolic Parsing** | Can parse `x * 0.88`. | Function logic depends on a string variable `status` found only in text. `x * ?` is unsolvable. |
| **Narrative Design** | Information-only. | **Authoritative**. The code logic branches based on narrative facts. |

This makes the code **Undecidable** without the Context.
