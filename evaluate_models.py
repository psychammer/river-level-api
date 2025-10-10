# evaluate_models.py
"""
Offline Evaluation Script for Water Level Forecasting Models.

This script loads the pre-trained 'full' and 'ablated' models and evaluates
their performance across the entire 'imputed_features.npy' dataset.

It calculates the average MAE, RMSE, and NSE for each forecast horizon
for the sole purpose of generating the final metrics to be hardcoded into the API.

To Run:
1. Make sure you have the required libraries (torch, numpy, etc.).
2. Place this file in the root of your project directory.
3. Run `python evaluate_models.py` from your terminal.
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import GCNConv, GATConv
import math
import os

# --- Configuration ---
# All paths point to the /models directory
class EvalConfig:
    MODEL_DIR = './models/'
    FEATURES_FILE = os.path.join(MODEL_DIR, 'imputed_features.npy')
    ADJ_FILE = os.path.join(MODEL_DIR, 'adjacency_matrix.npy')
    SCALER_MEAN_FILE = os.path.join(MODEL_DIR, 'scaler_mean.npy')
    SCALER_STD_FILE = os.path.join(MODEL_DIR, 'scaler_std.npy')

    # Model Paths
    FULL_MODEL_PATH = os.path.join(MODEL_DIR, 'full_forecaster_model.pth')
    ABLATED_MODEL_PATH = os.path.join(MODEL_DIR, 'ablated_forecaster_model.pth')
    ENCODER_PATH = os.path.join(MODEL_DIR, 'stgae_encoder.pth')

    # Data & Model Specs (must match training)
    NUM_STATIONS = 5
    NUM_FEATURES = 10
    TARGET_FEATURE_IDX = 0
    LOOKBACK_WINDOW = 24 * 3  # 3 days
    FORECAST_HORIZONS = [1, 3, 6, 12, 24, 48]
    
    # Hyperparameters (must match training)
    STGAE_GCN_HIDDEN = 64
    STGAE_GRU_HIDDEN_FACTOR = 64
    GAT_IN_FEATURES_FULL = STGAE_GCN_HIDDEN
    GAT_IN_FEATURES_ABLATED = NUM_FEATURES
    GAT_HIDDEN_DIM = 32
    GAT_HEADS = 4
    TRANSFORMER_D_MODEL = GAT_HIDDEN_DIM * GAT_HEADS
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 3
    TRANSFORMER_FF_DIM = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = EvalConfig()

# --- Model Definitions (Copied from training scripts) ---
# These are required for PyTorch to load the saved models correctly.

class FrozenSTGAEEncoder(nn.Module):
    def __init__(self, in_features, gcn_hidden, gru_hidden, encoder_weights_path):
        super().__init__()
        self.encoder_gcn = GCNConv(in_features, gcn_hidden)
        self.encoder_gru = nn.GRU(gcn_hidden * config.NUM_STATIONS, gru_hidden, batch_first=True)
        self.activation = nn.Tanh()
        self.load_encoder_weights(encoder_weights_path)
        for param in self.parameters(): param.requires_grad = False
    
    def load_encoder_weights(self, weights_path):
        encoder_state = torch.load(weights_path, map_location=config.DEVICE)
        self.encoder_gcn.load_state_dict(encoder_state['encoder_gcn'])
        self.encoder_gru.load_state_dict(encoder_state['encoder_gru'])
    
    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, _ = x.shape
        gcn_outputs = []
        for t in range(seq_len):
            xt = x[:, t, :, :].reshape(-1, x.size(3))
            edge_index_batched = edge_index.clone()
            for i in range(1, batch_size):
                edge_index_batched = torch.cat([edge_index_batched, edge_index + i * num_nodes], dim=1)
            gcn_out = self.activation(self.encoder_gcn(xt, edge_index_batched)).reshape(batch_size, num_nodes, -1)
            gcn_outputs.append(gcn_out)
        gcn_features = torch.stack(gcn_outputs, dim=1)
        gru_in = gcn_features.reshape(batch_size, seq_len, -1)
        _, temporal_encoding = self.encoder_gru(gru_in)
        return gcn_features, temporal_encoding.squeeze(0)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_dim, heads):
        super().__init__()
        self.gat = GATConv(in_features, hidden_dim, heads=heads, concat=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim * heads)
    def forward(self, x, edge_index):
        return self.norm(self.gat(x, edge_index))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(0), :]

class TemporalTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, ff_dim, max_seq_len):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        return self.transformer(x)

class MultiHorizonForecastHead(nn.Module):
    def __init__(self, input_dim, horizons, num_stations):
        super().__init__()
        self.forecast_heads = nn.ModuleDict({
            f'horizon_{h}': nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(input_dim // 2, num_stations))
            for h in horizons
        })
    def forward(self, x):
        return {h: self.forecast_heads[f'horizon_{h}'](x) for h in config.FORECAST_HORIZONS}

class STGAEGATTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stgae_encoder = FrozenSTGAEEncoder(config.NUM_FEATURES, config.STGAE_GCN_HIDDEN, config.STGAE_GRU_HIDDEN_FACTOR * config.NUM_STATIONS, config.ENCODER_PATH)
        self.gat = GraphAttentionLayer(config.GAT_IN_FEATURES_FULL, config.GAT_HIDDEN_DIM, config.GAT_HEADS)
        self.transformer = TemporalTransformer(config.TRANSFORMER_D_MODEL, config.TRANSFORMER_HEADS, config.TRANSFORMER_LAYERS, config.TRANSFORMER_FF_DIM, config.LOOKBACK_WINDOW)
        self.forecast_head = MultiHorizonForecastHead(config.TRANSFORMER_D_MODEL, config.FORECAST_HORIZONS, config.NUM_STATIONS)
    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, _ = x.shape
        with torch.no_grad(): gcn_features, _ = self.stgae_encoder(x, edge_index)
        gcn_flat = gcn_features.reshape(-1, gcn_features.size(-1))
        edge_index_batched = edge_index.clone()
        for i in range(1, batch_size * seq_len): edge_index_batched = torch.cat([edge_index_batched, edge_index + i * num_nodes], dim=1)
        gat_out = self.gat(gcn_flat, edge_index_batched)
        gat_features = gat_out.reshape(batch_size, seq_len, num_nodes, -1)
        transformer_input = gat_features.mean(dim=2)
        transformer_out = self.transformer(transformer_input)
        return self.forecast_head(transformer_out[:, -1, :])

class AblatedGATTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gat = GraphAttentionLayer(config.GAT_IN_FEATURES_ABLATED, config.GAT_HIDDEN_DIM, config.GAT_HEADS)
        self.transformer = TemporalTransformer(config.TRANSFORMER_D_MODEL, config.TRANSFORMER_HEADS, config.TRANSFORMER_LAYERS, config.TRANSFORMER_FF_DIM, config.LOOKBACK_WINDOW)
        self.forecast_head = MultiHorizonForecastHead(config.TRANSFORMER_D_MODEL, config.FORECAST_HORIZONS, config.NUM_STATIONS)
    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, num_features = x.shape
        x_flat = x.reshape(-1, num_features)
        edge_index_batched = edge_index.clone()
        for i in range(1, batch_size * seq_len): edge_index_batched = torch.cat([edge_index_batched, edge_index + i * num_nodes], dim=1)
        gat_out = self.gat(x_flat, edge_index_batched)
        gat_features = gat_out.reshape(batch_size, seq_len, num_nodes, -1)
        transformer_input = gat_features.mean(dim=2)
        transformer_out = self.transformer(transformer_input)
        return self.forecast_head(transformer_out[:, -1, :])

# --- Metrics Calculation ---
class PerformanceMetrics:
    @staticmethod
    def calculate_mae(y_true, y_pred): return np.nanmean(np.abs(y_pred - y_true))
    @staticmethod
    def calculate_rmse(y_true, y_pred): return np.sqrt(np.nanmean((y_pred - y_true)**2))
    @staticmethod
    def calculate_nse(y_true, y_pred):
        mask = ~np.isnan(y_true)
        if not np.any(mask): return np.nan
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        numerator = np.sum((y_true_masked - y_pred_masked)**2)
        denominator = np.sum((y_true_masked - np.mean(y_true_masked))**2)
        return 1 - (numerator / denominator) if denominator != 0 else np.nan

# --- Main Evaluation Function ---
def run_evaluation():
    print(f"--- Starting Offline Model Evaluation on {config.DEVICE} ---")
    
    # 1. Load data, scalers, and adjacency matrix
    print("Loading data and supporting files...")
    features = np.load(config.FEATURES_FILE)
    adj_matrix = np.load(config.ADJ_FILE)
    mean = np.load(config.SCALER_MEAN_FILE)
    std = np.load(config.SCALER_STD_FILE)
    edge_index = torch.tensor(np.where(adj_matrix > 0), dtype=torch.long).to(config.DEVICE)
    
    # 2. Load trained models
    full_model = STGAEGATTransformer(config).to(config.DEVICE)
    full_model.load_state_dict(torch.load(config.FULL_MODEL_PATH, map_location=config.DEVICE))
    full_model.eval()
    print("✅ Full model loaded successfully.")
    
    ablated_model = AblatedGATTransformer(config).to(config.DEVICE)
    ablated_model.load_state_dict(torch.load(config.ABLATED_MODEL_PATH, map_location=config.DEVICE))
    ablated_model.eval()
    print("✅ Ablated model loaded successfully.")

    # 3. Scale the data
    scaled_features = (features - mean) / (std + 1e-8)
    
    # 4. Perform sliding window evaluation
    print(f"\nRunning evaluation with a sliding window across {len(features)} timesteps...")
    
    # This dictionary will store the metric values for every single prediction
    all_metrics = {
        'full': {h: {'mae': [], 'rmse': [], 'nse': []} for h in config.FORECAST_HORIZONS},
        'ablated': {h: {'mae': [], 'rmse': [], 'nse': []} for h in config.FORECAST_HORIZONS}
    }

    # Define the range to iterate over, ensuring there's room for lookback and forecast
    max_horizon = max(config.FORECAST_HORIZONS)
    num_predictions = len(scaled_features) - config.LOOKBACK_WINDOW - max_horizon
    
    for i in tqdm(range(num_predictions), desc="Evaluating"):
        input_start = i
        input_end = i + config.LOOKBACK_WINDOW
        
        # Prepare model input
        input_window = scaled_features[input_start:input_end]
        input_tensor = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        
        with torch.no_grad():
            full_preds_scaled = full_model(torch.nan_to_num(input_tensor), edge_index)
            ablated_preds_scaled = ablated_model(torch.nan_to_num(input_tensor), edge_index)

        # Evaluate each horizon
        for h in config.FORECAST_HORIZONS:
            truth_idx = input_end + h - 1
            
            # Get scaled ground truth for the target feature (water level)
            actual_scaled = scaled_features[truth_idx, :, config.TARGET_FEATURE_IDX]
            
            # Inverse scale for meaningful metrics
            wl_mean = mean[config.TARGET_FEATURE_IDX]
            wl_std = std[config.TARGET_FEATURE_IDX]
            
            actual_values = (actual_scaled * wl_std) + wl_mean
            
            full_pred_values = (full_preds_scaled[h].squeeze().cpu().numpy() * wl_std) + wl_mean
            ablated_pred_values = (ablated_preds_scaled[h].squeeze().cpu().numpy() * wl_std) + wl_mean
            
            # Calculate and append metrics
            all_metrics['full'][h]['mae'].append(PerformanceMetrics.calculate_mae(actual_values, full_pred_values))
            all_metrics['full'][h]['rmse'].append(PerformanceMetrics.calculate_rmse(actual_values, full_pred_values))
            all_metrics['full'][h]['nse'].append(PerformanceMetrics.calculate_nse(actual_values, full_pred_values))

            all_metrics['ablated'][h]['mae'].append(PerformanceMetrics.calculate_mae(actual_values, ablated_pred_values))
            all_metrics['ablated'][h]['rmse'].append(PerformanceMetrics.calculate_rmse(actual_values, ablated_pred_values))
            all_metrics['ablated'][h]['nse'].append(PerformanceMetrics.calculate_nse(actual_values, ablated_pred_values))

    # 5. Average and print the final results
    print("\n\n" + "="*60)
    print("--- FINAL AVERAGED PERFORMANCE METRICS ---")
    print("="*60)
    
    for model_name, metrics_data in all_metrics.items():
        print(f"\n📊 Results for: {model_name.upper()} Model")
        print("   Horizon | Avg. MAE | Avg. RMSE | Avg. NSE")
        print("   " + "-"*45)
        for h in config.FORECAST_HORIZONS:
            avg_mae = np.nanmean(metrics_data[h]['mae'])
            avg_rmse = np.nanmean(metrics_data[h]['rmse'])
            avg_nse = np.nanmean(metrics_data[h]['nse'])
            print(f"   {h:<7}h | {avg_mae:8.4f} | {avg_rmse:9.4f} | {avg_nse:8.4f}")
    
    print("\n" + "="*60)
    print("Evaluation complete. You can now copy these values into your API.")

if __name__ == "__main__":
    run_evaluation()