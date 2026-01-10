use anyhow::Result;
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
    client: Client,
    challenge_url: String,
}

impl Validator {
    async fn new(args: Args) -> Result<Self> {
        // Create HTTP client
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;
        
        info!("Validator initialized");
        info!("Challenge URL: {}", args.challenge_url);
        
        Ok(Self {
            args,
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

        // For now, use a fixed list of miners
        // In production, this would come from the validator API
        let miners = vec![
            MinerInfo {
                hotkey: "miner_hotkey_1".to_string(),
                model_endpoint: "http://localhost:8000/generate".to_string(),
                model_api_key: None,
            },
        ];

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

        // Log scores (in production, submit to validator API)
        if !scores.is_empty() {
            info!("Evaluation complete:");
            for (hotkey, score) in &scores {
                info!("  {}: {:.4}", hotkey, score);
            }
        }

        Ok(())
    }

    async fn run(&self) -> Result<()> {
        info!("Starting validator node...");
        info!("  NetUID: {}", self.args.netuid);
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
