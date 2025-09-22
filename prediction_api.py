# prediction_api.py
"""
Water Level Forecasting API
Deploys the trained STGAE-GAT-Transformer model for real-time predictions
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import requests
from datetime import datetime, timedelta
import math
import logging
import uvicorn
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    # Model paths (adjust these to your deployed environment)
    MODEL_DIR = './models/'
    STGAE_MODEL_PATH = MODEL_DIR + 'stgae_full_model.pth'
    ENCODER_PATH = MODEL_DIR + 'stgae_encoder.pth'
    FULL_MODEL_PATH = MODEL_DIR + 'full_forecaster_model.pth'
    ABLATED_MODEL_PATH = MODEL_DIR + 'ablated_forecaster_model.pth'  # Added ablated model
    ADJ_MATRIX_PATH = MODEL_DIR + 'adjacency_matrix.npy'
    SCALER_MEAN_PATH = MODEL_DIR + 'scaler_mean.npy'
    SCALER_STD_PATH = MODEL_DIR + 'scaler_std.npy'
    
    # Model specifications
    NUM_STATIONS = 5
    NUM_FEATURES = 10
    TARGET_FEATURE_IDX = 0  # water level
    LOOKBACK_WINDOW = 24 * 3  # 3 days
    FORECAST_HORIZONS = [1, 3, 6, 12, 24, 48]
    
    # Station configuration
    STATION_CONFIG = {
        'MONTALBAN': {
            'coords': (14.733083, 121.130580),
            'obscd': '11102202'
        },
        'NANGKA': {
            'coords': (14.674022, 121.109319),
            'obscd': '11103202'
        },
        'SAN MATEO': {
            'coords': (14.679547, 121.109733),
            'obscd': '11104202'
        },
        'STO NINO': {
            'coords': (14.635941, 121.093122),
            'obscd': '11105202'
        },
        'TUMANA': {
            'coords': (14.656427, 121.096508),
            'obscd': '11106202'
        }
    }
    
    STATION_ORDER = ['MONTALBAN', 'NANGKA', 'SAN MATEO', 'STO NINO', 'TUMANA']
    FEATURE_ORDER = [
        'waterlevel',           
        'temperature_2m',        
        'relative_humidity_2m',  
        'dew_point_2m',         
        'rain',        
        'wind_speed_10m',       
        'surface_pressure',     
        'wind_gusts_10m',       
        'wind_direction_10m',   
        'cloud_cover'           
    ]
    
    # Model hyperparameters (must match training)
    STGAE_GCN_HIDDEN = 64
    STGAE_GRU_HIDDEN_FACTOR = 64
    GAT_IN_FEATURES = STGAE_GCN_HIDDEN
    GAT_HIDDEN_DIM = 32
    GAT_HEADS = 4
    TRANSFORMER_D_MODEL = GAT_HIDDEN_DIM * GAT_HEADS
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 3
    TRANSFORMER_FF_DIM = 256
    
    # API endpoints
    WATER_API_BASE = "http://121.58.193.173:8080/water/map_list.do"
    WEATHER_API_BASE = "https://api.open-meteo.com/v1/forecast"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# --- Model Definitions (from your training code) ---

class STGAE(nn.Module):
    def __init__(self, in_features, gcn_hidden, gru_hidden):
        super().__init__()
        self.encoder_gcn = GCNConv(in_features, gcn_hidden)
        self.encoder_gru = nn.GRU(gcn_hidden * config.NUM_STATIONS, gru_hidden, batch_first=True)
        
        self.decoder_gru = nn.GRU(gru_hidden, gru_hidden, batch_first=True)
        self.decoder_fc = nn.Linear(gru_hidden, gcn_hidden * config.NUM_STATIONS)
        self.decoder_gcn = GCNConv(gcn_hidden, in_features)
        
        self.relu = nn.Tanh()

    def forward(self, x, edge_index):
        x_original = x.clone()
        batch_size, seq_len, num_nodes, _ = x.shape
        
        gcn_outputs = []
        for t in range(seq_len):
            xt = x[:, t, :, :]
            xt_reshaped = xt.reshape(-1, x.size(3))
            
            edge_index_batched = edge_index.clone()
            for i in range(1, batch_size):
                edge_index_batched = torch.cat(
                    [edge_index_batched, edge_index + i * num_nodes], dim=1
                )

            gcn_out = self.relu(self.encoder_gcn(xt_reshaped, edge_index_batched))
            gcn_out_reshaped = gcn_out.reshape(batch_size, num_nodes, -1)
            gcn_outputs.append(gcn_out_reshaped)
            
        gcn_sequence = torch.stack(gcn_outputs, dim=1)
        gru_in = gcn_sequence.reshape(batch_size, seq_len, -1)
        _, latent_rep = self.encoder_gru(gru_in) 
        latent_rep = latent_rep.squeeze(0)

        decoder_gru_in = latent_rep.unsqueeze(1).repeat(1, seq_len, 1)
        gru_out, _ = self.decoder_gru(decoder_gru_in)
        fc_out = self.relu(self.decoder_fc(gru_out))
        
        decoder_outputs = []
        for t in range(seq_len):
            fct = fc_out[:, t, :]
            fct_reshaped = fct.reshape(-1, self.encoder_gcn.out_channels)

            edge_index_batched = edge_index.clone()
            for i in range(1, batch_size):
                edge_index_batched = torch.cat(
                    [edge_index_batched, edge_index + i * num_nodes], dim=1
                )

            recon_t = self.decoder_gcn(fct_reshaped, edge_index_batched)
            recon_t_reshaped = recon_t.reshape(batch_size, num_nodes, -1)
            decoder_outputs.append(recon_t_reshaped)
            
        reconstruction = torch.stack(decoder_outputs, dim=1)
        reconstruction = reconstruction + x_original
        return reconstruction

class FrozenSTGAEEncoder(nn.Module):
    def __init__(self, in_features, gcn_hidden, gru_hidden, encoder_weights_path):
        super().__init__()
        self.encoder_gcn = GCNConv(in_features, gcn_hidden)
        self.encoder_gru = nn.GRU(gcn_hidden * config.NUM_STATIONS, gru_hidden, batch_first=True)
        self.activation = nn.Tanh()
        self.load_encoder_weights(encoder_weights_path)
        for param in self.parameters():
            param.requires_grad = False
    
    def load_encoder_weights(self, weights_path):
        encoder_state = torch.load(weights_path, map_location=config.DEVICE, weights_only=True)
        self.encoder_gcn.load_state_dict(encoder_state['encoder_gcn'])
        self.encoder_gru.load_state_dict(encoder_state['encoder_gru'])
    
    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, _ = x.shape
        
        gcn_outputs = []
        for t in range(seq_len):
            xt = x[:, t, :, :]
            xt_reshaped = xt.reshape(-1, x.size(3))
            
            edge_index_batched = edge_index.clone()
            for i in range(1, batch_size):
                edge_index_batched = torch.cat(
                    [edge_index_batched, edge_index + i * num_nodes], dim=1
                )
            
            gcn_out = self.activation(self.encoder_gcn(xt_reshaped, edge_index_batched))
            gcn_out_reshaped = gcn_out.reshape(batch_size, num_nodes, -1)
            gcn_outputs.append(gcn_out_reshaped)
        
        gcn_features = torch.stack(gcn_outputs, dim=1)
        gru_in = gcn_features.reshape(batch_size, seq_len, -1)
        _, temporal_encoding = self.encoder_gru(gru_in)
        temporal_encoding = temporal_encoding.squeeze(0)
        
        return gcn_features, temporal_encoding

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_dim, heads):
        super().__init__()
        self.gat = GATConv(in_features, hidden_dim, heads=heads, concat=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim * heads)
        
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.norm(x)
        return x

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

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TemporalTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, ff_dim, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        output = self.transformer(x)
        return output

class MultiHorizonForecastHead(nn.Module):
    def __init__(self, input_dim, horizons, num_stations, target_feature_idx):
        super().__init__()
        self.horizons = horizons
        self.num_stations = num_stations
        self.target_feature_idx = target_feature_idx
        
        self.forecast_heads = nn.ModuleDict({
            f'horizon_{h}': nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 2, num_stations)
            )
            for h in horizons
        })
        
    def forward(self, x):
        predictions = {}
        for horizon in self.horizons:
            pred = self.forecast_heads[f'horizon_{horizon}'](x)
            predictions[horizon] = pred
        return predictions

class STGAEGATTransformer(nn.Module):
    def __init__(self, config, encoder_weights_path):
        super().__init__()
        
        self.stgae_encoder = FrozenSTGAEEncoder(
            in_features=config.NUM_FEATURES,
            gcn_hidden=config.STGAE_GCN_HIDDEN,
            gru_hidden=config.STGAE_GRU_HIDDEN_FACTOR * config.NUM_STATIONS,
            encoder_weights_path=encoder_weights_path
        )
        
        self.gat = GraphAttentionLayer(
            in_features=config.GAT_IN_FEATURES,
            hidden_dim=config.GAT_HIDDEN_DIM,
            heads=config.GAT_HEADS
        )
        
        self.transformer = TemporalTransformer(
            d_model=config.TRANSFORMER_D_MODEL,
            n_heads=config.TRANSFORMER_HEADS,
            n_layers=config.TRANSFORMER_LAYERS,
            ff_dim=config.TRANSFORMER_FF_DIM,
            max_seq_len=config.LOOKBACK_WINDOW
        )
        
        self.forecast_head = MultiHorizonForecastHead(
            input_dim=config.TRANSFORMER_D_MODEL,
            horizons=config.FORECAST_HORIZONS,
            num_stations=config.NUM_STATIONS,
            target_feature_idx=config.TARGET_FEATURE_IDX
        )
        
    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        with torch.no_grad():
            gcn_features, temporal_encoding = self.stgae_encoder(x, edge_index)
        
        gcn_flat = gcn_features.reshape(-1, gcn_features.size(-1))
        
        edge_index_batched = edge_index.clone()
        for i in range(1, batch_size * seq_len):
            edge_index_batched = torch.cat(
                [edge_index_batched, edge_index + i * num_nodes], dim=1
            )
        
        gat_out = self.gat(gcn_flat, edge_index_batched)
        gat_features = gat_out.reshape(batch_size, seq_len, num_nodes, -1)
        transformer_input = gat_features.mean(dim=2)
        transformer_out = self.transformer(transformer_input)
        last_timestep = transformer_out[:, -1, :]
        predictions = self.forecast_head(last_timestep)
        
        return predictions

class AblatedGATTransformer(nn.Module):
    """
    Ablated model: Raw Features → GAT → Transformer → Multi-Horizon Forecasting
    (No STGAE preprocessing)
    """
    def __init__(self, config):
        super().__init__()
        
        # GAT takes raw features directly (NUM_FEATURES instead of STGAE_GCN_HIDDEN)
        self.gat = GraphAttentionLayer(
            in_features=config.NUM_FEATURES,  # Direct raw features
            hidden_dim=config.GAT_HIDDEN_DIM,
            heads=config.GAT_HEADS
        )
        
        self.transformer = TemporalTransformer(
            d_model=config.TRANSFORMER_D_MODEL,
            n_heads=config.TRANSFORMER_HEADS,
            n_layers=config.TRANSFORMER_LAYERS,
            ff_dim=config.TRANSFORMER_FF_DIM,
            max_seq_len=config.LOOKBACK_WINDOW
        )
        
        self.forecast_head = MultiHorizonForecastHead(
            input_dim=config.TRANSFORMER_D_MODEL,
            horizons=config.FORECAST_HORIZONS,
            num_stations=config.NUM_STATIONS,
            target_feature_idx=config.TARGET_FEATURE_IDX
        )
        
    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # Apply GAT directly to raw features
        x_flat = x.reshape(-1, num_features)
        
        # Create batched edge index for GAT
        edge_index_batched = edge_index.clone()
        for i in range(1, batch_size * seq_len):
            edge_index_batched = torch.cat(
                [edge_index_batched, edge_index + i * num_nodes], dim=1
            )
        
        # GAT forward (processes raw features)
        gat_out = self.gat(x_flat, edge_index_batched)
        
        # Reshape back
        gat_features = gat_out.reshape(batch_size, seq_len, num_nodes, -1)
        
        # Aggregate spatial features for transformer input
        transformer_input = gat_features.mean(dim=2)
        
        # Temporal modeling with Transformer
        transformer_out = self.transformer(transformer_input)
        
        # Use the last timestep for forecasting
        last_timestep = transformer_out[:, -1, :]
        
        # Multi-horizon forecasting
        predictions = self.forecast_head(last_timestep)
        
        return predictions

# --- Data Fetching and Processing ---

class DataFetcher:
    """Fetches and processes real-time data from APIs"""
    
    @staticmethod
    def fetch_water_levels(start_datetime: datetime, end_datetime: datetime) -> pd.DataFrame:
        """Fetch water level data for all stations"""
        logger.info(f"Fetching water levels from {start_datetime} to {end_datetime}")
        
        all_data = []
        current = start_datetime
        
        while current <= end_datetime:
            ymdhm = current.strftime('%Y%m%d%H%M')
            url = f"{config.WATER_API_BASE}?ymdhm={ymdhm}"
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                for station in data:
                    obscd = station.get('obscd')
                    station_name = None
                    for name, cfg in config.STATION_CONFIG.items():
                        if cfg['obscd'] == obscd:
                            station_name = name
                            break
                    
                    if station_name and station.get('wl'):
                        try:
                            wl_value = float(station['wl'])
                            all_data.append({
                                'datetime': current,
                                'station': station_name,
                                'waterlevel': wl_value
                            })
                        except (ValueError, TypeError):
                            pass
                            
            except Exception as e:
                logger.warning(f"Failed to fetch water data for {ymdhm}: {e}")
            
            current += timedelta(hours=1)
        
        if all_data:
            df = pd.DataFrame(all_data)
            df_pivot = df.pivot_table(
                index='datetime',
                columns='station',
                values='waterlevel'
            )
            return df_pivot
        else:
            return pd.DataFrame()
    
    @staticmethod
    def fetch_weather_data(station_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch weather data for a specific station"""
        coords = config.STATION_CONFIG[station_name]['coords']
        
        params = {
            'latitude': coords[0],
            'longitude': coords[1],
            'hourly': ','.join([
                'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
                'rain', 'wind_speed_10m', 'wind_gusts_10m', 'cloud_cover',
                'wind_direction_10m', 'surface_pressure'
            ]),
            'timezone': 'Asia/Singapore',
            'start_date': start_date,
            'end_date': end_date
        }
        
        try:
            response = requests.get(config.WEATHER_API_BASE, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data['hourly'])
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Rename columns to match feature order
            df.rename(columns={'rain': 'rain'}, inplace=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch weather for {station_name}: {e}")
            return pd.DataFrame()

class DataProcessor:
    """Processes raw data for model input"""
    
    def __init__(self):
        # Load scalers and adjacency matrix
        self.mean = np.load(config.SCALER_MEAN_PATH)
        self.std = np.load(config.SCALER_STD_PATH)
        self.adj_matrix = np.load(config.ADJ_MATRIX_PATH)
        self.edge_index = torch.tensor(
            np.array(np.where(self.adj_matrix > 0)), 
            dtype=torch.long
        ).to(config.DEVICE)
        
        # Load STGAE for imputation
        self.stgae_model = STGAE(
            in_features=config.NUM_FEATURES,
            gcn_hidden=config.STGAE_GCN_HIDDEN,
            gru_hidden=config.STGAE_GRU_HIDDEN_FACTOR * config.NUM_STATIONS
        ).to(config.DEVICE)
        
        stgae_state = torch.load(config.STGAE_MODEL_PATH, map_location=config.DEVICE, weights_only=True)
        self.stgae_model.load_state_dict(stgae_state)
        self.stgae_model.eval()
    
    def prepare_features(self, water_df: pd.DataFrame, weather_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Combine water and weather data into feature array"""
        
        # Create a complete time index
        time_index = water_df.index
        
        # Initialize feature array
        num_timesteps = len(time_index)
        feature_array = np.full(
            (num_timesteps, config.NUM_STATIONS, config.NUM_FEATURES),
            np.nan,
            dtype=np.float32
        )
        
        # Fill in water levels
        for s_idx, station in enumerate(config.STATION_ORDER):
            if station in water_df.columns:
                feature_array[:, s_idx, 0] = water_df[station].values
        
        # Fill in weather features
        for s_idx, station in enumerate(config.STATION_ORDER):
            if station in weather_data and not weather_data[station].empty:
                weather_df = weather_data[station].reindex(time_index)
                
                for f_idx, feature in enumerate(config.FEATURE_ORDER[1:], start=1):
                    if feature in weather_df.columns:
                        feature_array[:, s_idx, f_idx] = weather_df[feature].values
        
        return feature_array
    
    def impute_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Use STGAE to impute missing values"""
        if features.shape[0] < config.LOOKBACK_WINDOW:
            # Pad with NaNs if not enough data
            pad_size = config.LOOKBACK_WINDOW - features.shape[0]
            padding = np.full((pad_size, features.shape[1], features.shape[2]), np.nan)
            features = np.vstack([padding, features])
        
        # Scale features
        scaled_features = (features - self.mean) / (self.std + 1e-8)
        
        # Prepare input tensor
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        input_clean = torch.nan_to_num(input_tensor)
        
        # Run STGAE for imputation
        with torch.no_grad():
            reconstruction = self.stgae_model(input_clean, self.edge_index)
        
        # Fill NaN values with reconstructed values
        reconstruction_np = reconstruction.squeeze(0).cpu().numpy()
        scaled_features_imputed = scaled_features.copy()
        nan_mask = np.isnan(scaled_features)
        scaled_features_imputed[nan_mask] = reconstruction_np[nan_mask]
        
        # Inverse scale
        features_imputed = (scaled_features_imputed * (self.std + 1e-8)) + self.mean
        
        # Return only the lookback window
        return features_imputed[-config.LOOKBACK_WINDOW:]

# --- Model Manager ---

class ModelManager:
    """Manages both full and ablated forecasting models"""
    
    def __init__(self):
        # Load both models
        self.full_model = STGAEGATTransformer(config, config.ENCODER_PATH).to(config.DEVICE)
        self.ablated_model = AblatedGATTransformer(config).to(config.DEVICE)
        
        # Load trained weights for full model
        try:
            full_state_dict = torch.load(config.FULL_MODEL_PATH, map_location=config.DEVICE, weights_only=True)
            self.full_model.load_state_dict(full_state_dict)
            self.full_model.eval()
            logger.info("Full model (with STGAE) loaded successfully")
        except FileNotFoundError:
            logger.warning(f"Full model not found at {config.FULL_MODEL_PATH}")
            self.full_model = None
        
        # Load trained weights for ablated model
        try:
            ablated_state_dict = torch.load(config.ABLATED_MODEL_PATH, map_location=config.DEVICE, weights_only=True)
            self.ablated_model.load_state_dict(ablated_state_dict)
            self.ablated_model.eval()
            logger.info("Ablated model (without STGAE) loaded successfully")
        except FileNotFoundError:
            logger.warning(f"Ablated model not found at {config.ABLATED_MODEL_PATH}")
            self.ablated_model = None
        
        self.processor = DataProcessor()
        
        # Check which models are available
        self.available_models = []
        if self.full_model is not None:
            self.available_models.append("full")
        if self.ablated_model is not None:
            self.available_models.append("ablated")
        
        if not self.available_models:
            raise RuntimeError("No models available! Please ensure model files exist.")
        
        logger.info(f"Available models: {self.available_models}")
    
    def predict(self, features: np.ndarray, model_type: str = "full") -> Dict[int, np.ndarray]:
        """
        Make predictions for all forecast horizons
        
        Args:
            features: Input feature array
            model_type: "full" for STGAE-GAT-Transformer, "ablated" for GAT-Transformer only
        """
        
        # Validate model type
        if model_type not in self.available_models:
            raise ValueError(f"Model type '{model_type}' not available. Choose from: {self.available_models}")
        
        # Impute missing values
        features_imputed = self.processor.impute_missing_values(features)
        
        # Scale features
        features_scaled = (features_imputed - self.processor.mean) / (self.processor.std + 1e-8)
        
        # Prepare tensor
        input_tensor = torch.tensor(
            features_scaled, 
            dtype=torch.float32
        ).unsqueeze(0).to(config.DEVICE)
        
        # Select model
        model = self.full_model if model_type == "full" else self.ablated_model
        
        # Make predictions
        with torch.no_grad():
            predictions_scaled = model(
                torch.nan_to_num(input_tensor),
                self.processor.edge_index
            )
        
        # Inverse scale predictions (only for water level)
        predictions = {}
        water_mean = self.processor.mean[0]
        water_std = self.processor.std[0]
        
        for horizon in config.FORECAST_HORIZONS:
            pred_scaled = predictions_scaled[horizon].cpu().numpy()
            pred_original = (pred_scaled * water_std) + water_mean
            predictions[horizon] = pred_original.squeeze()
        
        return predictions

# --- API Definition ---

app = FastAPI(title="Water Level Forecasting API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_manager
    model_manager = ModelManager()
    logger.info("API started successfully")

# Request/Response models
class PredictionRequest(BaseModel):
    lookback_hours: Optional[int] = 72  # Default to 3 days
    model_type: Optional[str] = "full"  # "full" or "ablated"

class StationPrediction(BaseModel):
    station: str
    current_level: Optional[float]
    predictions: Dict[str, float]
    alert_levels: Dict[str, float]

class PredictionResponse(BaseModel):
    timestamp: str
    model_type: str
    data_range: Dict[str, str]
    predictions: List[StationPrediction]

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online", 
        "models": {
            "available": model_manager.available_models if model_manager else [],
            "full": "STGAE-GAT-Transformer",
            "ablated": "GAT-Transformer (no STGAE)"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_water_levels(request: PredictionRequest):
    """
    Predict water levels for all stations at multiple horizons
    
    Args:
        lookback_hours: Number of hours of historical data to use
        model_type: "full" (with STGAE) or "ablated" (without STGAE)
    """
    try:
        # Validate model type
        if request.model_type not in ["full", "ablated"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model_type. Choose 'full' or 'ablated'"
            )
        
        if request.model_type not in model_manager.available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_type}' not available. Available models: {model_manager.available_models}"
            )
        
        # Calculate time range
        now = datetime.now()
        end_time = now.replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=request.lookback_hours)
        
        logger.info(f"Fetching data from {start_time} to {end_time}")
        logger.info(f"Using model: {request.model_type}")
        
        # Fetch water level data
        fetcher = DataFetcher()
        water_df = fetcher.fetch_water_levels(start_time, end_time)
        
        if water_df.empty:
            raise HTTPException(status_code=404, detail="No water level data available")
        
        # Fetch weather data for all stations
        weather_data = {}
        start_date = start_time.strftime('%Y-%m-%d')
        end_date = end_time.strftime('%Y-%m-%d')
        
        for station in config.STATION_ORDER:
            weather_df = fetcher.fetch_weather_data(station, start_date, end_date)
            weather_data[station] = weather_df
        
        # Prepare features
        features = model_manager.processor.prepare_features(water_df, weather_data)
        
        # Make predictions with selected model
        predictions_raw = model_manager.predict(features, model_type=request.model_type)
        
        # Format response
        response_data = []
        
        for s_idx, station in enumerate(config.STATION_ORDER):
            # Get current water level
            current_level = None
            if station in water_df.columns:
                last_values = water_df[station].dropna()
                if not last_values.empty:
                    current_level = float(last_values.iloc[-1])
            
            # Format predictions
            station_predictions = {}
            for horizon in config.FORECAST_HORIZONS:
                pred_value = float(predictions_raw[horizon][s_idx])
                station_predictions[f"{horizon}h"] = round(pred_value, 2)
            
            # Get alert levels (these would be from your database/config)
            # You should update these with actual values for each station
            alert_levels = {
                "alert": 22.4 if station == "MONTALBAN" else 16.5,
                "alarm": 23.0 if station == "MONTALBAN" else 17.1,
                "critical": 23.6 if station == "MONTALBAN" else 17.7
            }
            
            response_data.append(StationPrediction(
                station=station,
                current_level=current_level,
                predictions=station_predictions,
                alert_levels=alert_levels
            ))
        
        return PredictionResponse(
            timestamp=datetime.now().isoformat(),
            model_type=request.model_type,
            data_range={
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            predictions=response_data
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stations")
async def get_stations():
    """Get list of available stations"""
    stations = []
    for name, cfg in config.STATION_CONFIG.items():
        stations.append({
            "name": name,
            "coordinates": {
                "lat": cfg['coords'][0],
                "lon": cfg['coords'][1]
            },
            "obscd": cfg['obscd']
        })
    return {"stations": stations}

@app.get("/model_info")
async def get_model_info():
    """Get information about the models"""
    return {
        "available_models": model_manager.available_models if model_manager else [],
        "models": {
            "full": {
                "name": "STGAE-GAT-Transformer",
                "description": "Full model with STGAE encoder for feature extraction",
                "components": ["STGAE Encoder", "GAT", "Transformer", "Multi-Horizon Head"]
            },
            "ablated": {
                "name": "GAT-Transformer",
                "description": "Ablated model without STGAE preprocessing",
                "components": ["GAT", "Transformer", "Multi-Horizon Head"]
            }
        },
        "features": config.FEATURE_ORDER,
        "forecast_horizons": config.FORECAST_HORIZONS,
        "lookback_window": config.LOOKBACK_WINDOW,
        "num_stations": config.NUM_STATIONS,
        "device": config.DEVICE
    }

@app.post("/compare_models")
async def compare_models(request: PredictionRequest):
    """
    Compare predictions from both models
    """
    try:
        if not all(m in model_manager.available_models for m in ["full", "ablated"]):
            raise HTTPException(
                status_code=400,
                detail="Both models must be available for comparison"
            )
        
        # Calculate time range
        now = datetime.now()
        end_time = now.replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=request.lookback_hours)
        
        # Fetch data
        fetcher = DataFetcher()
        water_df = fetcher.fetch_water_levels(start_time, end_time)
        
        if water_df.empty:
            raise HTTPException(status_code=404, detail="No water level data available")
        
        weather_data = {}
        start_date = start_time.strftime('%Y-%m-%d')
        end_date = end_time.strftime('%Y-%m-%d')
        
        for station in config.STATION_ORDER:
            weather_df = fetcher.fetch_weather_data(station, start_date, end_date)
            weather_data[station] = weather_df
        
        # Prepare features
        features = model_manager.processor.prepare_features(water_df, weather_data)
        
        # Get predictions from both models
        full_predictions = model_manager.predict(features, model_type="full")
        ablated_predictions = model_manager.predict(features, model_type="ablated")
        
        # Format comparison
        comparison = {}
        for s_idx, station in enumerate(config.STATION_ORDER):
            station_comparison = {}
            for horizon in config.FORECAST_HORIZONS:
                full_pred = float(full_predictions[horizon][s_idx])
                ablated_pred = float(ablated_predictions[horizon][s_idx])
                station_comparison[f"{horizon}h"] = {
                    "full_model": round(full_pred, 2),
                    "ablated_model": round(ablated_pred, 2),
                    "difference": round(full_pred - ablated_pred, 2),
                    "percent_diff": round(((full_pred - ablated_pred) / ablated_pred * 100) if ablated_pred != 0 else 0, 1)
                }
            comparison[station] = station_comparison
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "comparison": comparison
        }
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)