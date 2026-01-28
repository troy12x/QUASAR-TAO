# Real Miner & Validator Testing Guide

## Current Status

✅ **API Server**: Running on http://localhost:8000  
✅ **Validator**: Running (quasar_validator)  
⏳ **Miner**: Ready to start (quasar_miner)

## Step-by-Step Testing

### Step 1: Start the Miner

In a new terminal, run:
```bash
cd /root/QUASAR-SUBNET
./START_MINER.sh
```

**Expected Output:**
- Miner loads model
- Gets current round
- Starts optimization loop
- Generates kernel code
- Forks repository
- Submits to API

### Step 2: Monitor Logs

Watch for these key events:

#### Miner Logs (should show):
```
[MINER] Getting current round...
[MINER] Generating optimized kernel...
[MINER] Forking repository...
[MINER] Committing code...
[MINER] Testing kernel performance...
[MINER] Submitting to validator API...
[MINER] Submission successful!
```

#### Validator Logs (should show):
```
[VALIDATOR] Checking for pending submissions...
[VALIDATOR] Found submission: <submission_id>
[VALIDATOR] Cloning repository...
[VALIDATOR] Running performance test...
[VALIDATOR] Submission validated: score=X.XX
[VALIDATOR] Marking as validated...
```

#### API Server Logs (should show):
```
POST /submit_kernel - 200 OK
Submission received: miner_hotkey=5G1c4wkiwKMZzQugGv3c3SoRdhsgHJAGXZmr5u1yeFP9k48h
```

### Step 3: Verify Submission

Check if submission was received:
```bash
# Check API for submissions
curl http://localhost:8000/get_current_round
```

### Step 4: Check Validation

The validator should automatically:
1. Poll for pending submissions (every 1-5 minutes based on activity)
2. Clone the miner's repository
3. Run performance tests
4. Mark submission as validated
5. Record success/failure for IP tracking

### Step 5: Monitor Round Finalization

When a round ends (or manually finalize):
```bash
# Get current round ID first
curl http://localhost:8000/get_current_round

# Finalize round (replace ROUND_ID)
curl -X POST http://localhost:8000/finalize_round/ROUND_ID
```

**Expected Output:**
- Round status changes to "completed"
- Rankings calculated
- Weights distributed (60/25/10/5)
- Baseline set for next round

## Troubleshooting

### Miner Issues

**Problem**: Miner not submitting
- Check GITHUB_TOKEN is set
- Check GITHUB_USERNAME is set
- Verify API server is running
- Check miner logs for errors

**Problem**: CUDA out of memory
- Reduce TARGET_SEQUENCE_LENGTH
- Reduce AGENT_ITERATIONS
- Check GPU memory usage

### Validator Issues

**Problem**: Validator not validating
- Check API server is accessible
- Verify VALIDATOR_API_URL is correct
- Check validator logs for connection errors
- Ensure validator has access to GPU

### API Issues

**Problem**: 500 errors
- Check database connection (Supabase)
- Verify .env file is loaded
- Check API server logs for errors

## Success Criteria

✅ Miner successfully submits kernel  
✅ Validator validates submission  
✅ Submission appears in database  
✅ Round can be finalized  
✅ Weights distributed correctly (60/25/10/5)  
✅ Baseline set for next round  

## Next Steps After Testing

1. Monitor for multiple rounds
2. Test with multiple miners
3. Verify IP banning works
4. Test dynamic polling
5. Verify baseline comparison (Round 2+)

## Log Locations

- **API Server**: Terminal running `./START_SERVER.sh`
- **Validator**: Terminal running `./START_VALIDATOR.sh`
- **Miner**: Terminal running `./START_MINER.sh`

## Quick Commands

```bash
# Check current round
curl http://localhost:8000/get_current_round

# Check submission rate
curl http://localhost:8000/get_submission_rate

# Get weights
curl http://localhost:8000/get_weights

# Check database directly
python scripts/test_db_connection.py
```
