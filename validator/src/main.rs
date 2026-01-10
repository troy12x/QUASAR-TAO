use anyhow::Result;
use bittensor::{subtensor::Subtensor, Keypair};
use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{error, info, warn};
use tracing_subscriber;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Subnet UID
    #[arg(short, long, default_value_t = 24)]
    netuid: u16,

    /// Wallet name
    #[arg(long, default_value = "validator")]
    wallet_name: String,

    /// Hotkey name
    #[arg(long, default_value = "default")]
    hotkey: String,

    /// Challenge container URL
    #[arg(long, default_value = "http://localhost:8080")]
    challenge_url: String,

    /// Subtensor network
    #[arg(long, default_value = "finney")]
    network: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct EvaluationRequest {
    request_id: String,
    submission_id: String,
    participant_id: String,
    data: serde_json::Value,
    metadata: serde_json::Value,
    epoch: u64,
    deadline: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct EvaluationResponse {
    request_id: String,
    success: bool,
    error: Option<String>,
    score: f64,
    results: serde_json::Value,
    execution_time_ms: i64,
    cost: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MinerInfo {
    hotkey: String,
    model_endpoint: String,
    model_api_key: Option<String>,
}

struct Validator {
    args: Args,
    keypair: Keypair,
    subtensor: Subtensor,
    client: Client,
    challenge_url: String,
}

impl Validator {
    async fn new(args: Args) -> Result<Self> {
        // Load keypair from wallet
        info!("Loading wallet: {}/{}", args.wallet_name, args.hotkey);
        // Note: In production, load from Bittensor wallet directory
        let keypair = Keypair::from_mnemonic("bottom metal radar abuse cool bamboo agent reveal fever bachelor way ranch")?;
        
        // Connect to subtensor
        info!("Connecting to subtensor network: {}", args.network);
        let subtensor = Subtensor::new(&args.network)?;
        
        // Create HTTP client
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;
        
        Ok(Self {
            args,
            keypair,
            subtensor,
            client,
            challenge_url: args.challenge_url,
        })
    }

    async fn evaluate_miner(&self, miner: &MinerInfo) -> Result<f64> {
        let request_id = uuid::Uuid::new_v4().to_string();
        
        let request = EvaluationRequest {
            request_id: request_id.clone(),
            submission_id: miner.hotkey.clone(),
            participant_id: miner.hotkey.clone(),
            data: serde_json::json!({
                "model_endpoint": miner.model_endpoint,
                "model_api_key": miner.model_api_key
            }),
            metadata: serde_json::json!({}),
            epoch: 0,
            deadline: 0,
        };

        info!("Evaluating miner: {}", miner.hotkey);
        info!("  Model endpoint: {}", miner.model_endpoint);

        let url = format!("{}/evaluate", self.challenge_url);
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            anyhow::bail!("Challenge container returned {}: {}", status, body);
        }

        let eval_response: EvaluationResponse = response.json().await?;
        
        if !eval_response.success {
            anyhow::bail!("Evaluation failed: {:?}", eval_response.error);
        }

        info!("  Score: {:.4}", eval_response.score);
        info!("  Execution time: {}ms", eval_response.execution_time_ms);

        Ok(eval_response.score)
    }

    async fn run_epoch(&self) -> Result<()> {
        info!("Starting epoch evaluation...");

        // Get metagraph to find miners
        let metagraph = self.subtensor.metagraph(self.args.netuid)?;
        info!("Found {} neurons in metagraph", metagraph.n);

        // Collect miners (neurons with stake < 1000 are miners)
        let mut miners = Vec::new();
        for (i, hotkey) in metagraph.hotkeys.iter().enumerate() {
            let stake = metagraph.S[i];
            // Miners have low stake or no stake
            if stake < 1000.0 {
                // In production, get model endpoint from miner registration
                // For now, use placeholder
                miners.push(MinerInfo {
                    hotkey: hotkey.clone(),
                    model_endpoint: format!("http://{}:8000/generate", hotkey),
                    model_api_key: None,
                });
            }
        }

        info!("Found {} miners to evaluate", miners.len());

        // Evaluate each miner
        let mut scores = Vec::new();
        for miner in miners {
            match self.evaluate_miner(&miner).await {
                Ok(score) => {
                    scores.push((miner.hotkey.clone(), score));
                }
                Err(e) => {
                    warn!("Failed to evaluate miner {}: {}", miner.hotkey, e);
                }
            }
        }

        // Submit weights to Bittensor
        if !scores.is_empty() {
            info!("Submitting weights for {} miners", scores.len());
            
            // Convert scores to weights
            let uids: Vec<u16> = metagraph.hotkeys
                .iter()
                .enumerate()
                .filter(|(_, h)| scores.iter().any(|(hotkey, _)| hotkey == *h))
                .map(|(i, _)| i as u16)
                .collect();
            
            let weights: Vec<f64> = scores
                .iter()
                .map(|(_, score)| *score)
                .collect();

            // Normalize weights
            let total: f64 = weights.iter().sum();
            let normalized_weights: Vec<f64> = weights
                .iter()
                .map(|w| w / total)
                .collect();

            // Submit weights
            // Note: In production, use proper Bittensor weight submission
            info!("Would submit weights:");
            for (uid, weight) in uids.iter().zip(normalized_weights.iter()) {
                info!("  UID {}: {:.4}", uid, weight);
            }
        }

        Ok(())
    }

    async fn run(&self) -> Result<()> {
        info!("Starting validator node...");
        info!("  NetUID: {}", self.args.netuid);
        info!("  Hotkey: {}", self.keypair.ss58_address());
        info!("  Challenge URL: {}", self.challenge_url);

        loop {
            info!("Starting new epoch...");
            
            match self.run_epoch().await {
                Ok(_) => {
                    info!("Epoch completed successfully");
                }
                Err(e) => {
                    error!("Epoch failed: {}", e);
                }
            }

            // Wait for next epoch (~72 minutes)
            info!("Waiting 72 minutes for next epoch...");
            sleep(Duration::from_secs(72 * 60)).await;
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Parse arguments
    let args = Args::parse();

    // Create validator
    let validator = Validator::new(args).await?;

    // Run validator
    validator.run().await?;

    Ok(())
}
