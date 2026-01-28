# QUASAR-SUBNET: Requirements Summary

## ✅ Confirmed Requirements

### 1. Reward Distribution
- **1st place:** 60%
- **2nd place:** 25%
- **3rd place:** 10%
- **4th place:** 5%
- **Total:** 100% distributed to top 4 miners

### 2. Round-Based Competition
- **Round Duration:** 2 days (48 hours)
- **Structure:**
  - Miners submit anytime during round
  - At deadline, all submissions evaluated
  - Top solution determined
  - Rewards distributed
- **Round 2+:** Uses previous winner as baseline
  - Submissions must beat baseline to qualify
  - Creates progressive improvement

### 3. First-Submission-Wins
- **Problem:** Solution leakage (someone copies winning solution)
- **Solution:** Timestamp-based tiebreaker
- **Logic:** For identical results, first submission (by `created_at`) wins
- **Sorting:** `weighted_score DESC, created_at ASC`

### 4. IP Banning
- **Threshold:** 5 consecutive failures
- **Ban Duration:** 24 hours
- **Purpose:** Prevent spam and malicious submissions
- **Tracking:** Per IP address, separate from hotkey rate limiting

### 5. Dynamic Validation Frequency
- **High Activity (>5 submissions/min):** Poll every 1 minute
- **Medium Activity (1-5 submissions/min):** Poll every 2 minutes
- **Low Activity (<1 submission/min):** Poll every 5 minutes
- **Benefit:** Efficient resource usage, timely validation

### 6. Sequence Length Testing
- **Flexible:** Miners choose target sequence length
- **Testing:** Both small (512, 1024, 2048) and large (target) lengths
- **League Multipliers:** Still apply based on target length
  - 100k: 0.5x
  - 200k: 0.75x
  - ...
  - 1M: 3.0x (highest)

## 🎯 Key Features to Implement

### Round Management
- Create rounds with 48-hour duration
- Track round status (active, evaluating, completed)
- Finalize rounds at deadline
- Store baseline submission for next round

### Solution Hash & Duplicate Detection
- Calculate hash of solution (tokens/sec + sequence length + benchmarks)
- Use for detecting identical results
- First submission with same hash wins

### Baseline Comparison
- Round 1: No baseline (all submissions eligible)
- Round 2+: Previous winner is baseline
- Filter out submissions that don't beat baseline
- Only rank submissions above baseline

### IP Tracking & Banning
- Track IP address for each submission
- Count consecutive failures per IP
- Ban after 5 failures for 24 hours
- Reset count on successful submission

### Dynamic Polling
- Monitor submission rate (submissions per minute)
- Adjust validator polling interval automatically
- Efficient resource usage

## 📊 Example Round Flow

### Round 1 (No Baseline)
```
Day 1-2: Miners submit optimized kernels
Day 2 (Deadline): 
  - Evaluate all submissions
  - Rank by weighted_score (tokens/sec * multiplier)
  - First-submission-wins for ties
  - Top 4: 60%, 25%, 10%, 5%
  - Winner becomes baseline for Round 2
```

### Round 2 (With Baseline)
```
Baseline: Round 1 winner (e.g., 500k tokens/sec @ 1M = 1.5M weighted)
Day 1-2: Miners submit new kernels
Day 2 (Deadline):
  - Filter: Only submissions beating baseline (1.5M weighted)
  - Rank remaining submissions
  - Top 4: 60%, 25%, 10%, 5%
  - New winner becomes baseline for Round 3
```

## 🔧 Implementation Priority

### Phase 1: Core Round System (CRITICAL)
- Database schema for rounds
- Round creation and management
- Deadline handling

### Phase 2: First-Submission-Wins (CRITICAL)
- Solution hash calculation
- Timestamp-based sorting
- Duplicate detection

### Phase 3: IP Banning (CRITICAL)
- IP tracking model
- Failure counting
- Ban enforcement

### Phase 4: Reward Distribution (CRITICAL)
- Update to 60%, 25%, 10%, 5%
- Round-based weight calculation

### Phase 5: Baseline System (CRITICAL)
- Baseline comparison logic
- Filter submissions below baseline

### Phase 6: Dynamic Validation (HIGH)
- Submission rate monitoring
- Adaptive polling intervals

## 📝 Next Steps

1. **Review** `UPDATED_IMPLEMENTATION_PLAN.md` for detailed code
2. **Start with Phase 1** - Database and round management
3. **Test incrementally** after each phase
4. **Deploy to testnet** for validation
5. **Deploy to mainnet** once tested

## ⚠️ Important Notes

- **Solution Hash:** Normalize to 2 decimal places to account for minor variations
- **IP Banning:** Separate from hotkey-based rate limiting (different purposes)
- **Baseline:** Only applies from Round 2 onwards
- **First-Wins:** Only applies to identical solutions (same hash)
- **Dynamic Polling:** Validators check submission rate every cycle

---

**Ready to implement!** See `UPDATED_IMPLEMENTATION_PLAN.md` for step-by-step code.
