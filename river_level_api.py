# river_level_api.py
"""
River Level Prediction API
Serves predictions from trained STGAE-GAT-Transformer models
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import requests
import math
from enum import Enum
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
class ModelType(str, Enum):
    FULL = "full"
    ABLATED = "ablated"

class Config:
    # Model paths (actual paths)
    MODEL_DIR = Path("./models/")
    FULL_MODEL_PATH = MODEL_DIR / "full_forecaster_model.pth"
    ABLATED_MODEL_PATH = MODEL_DIR / "ablated_forecaster_model.pth"
    STGAE_ENCODER_PATH = MODEL_DIR / "stgae_encoder.pth"
    STGAE_FULL_PATH = MODEL_DIR / "stgae_full_model.pth"
    ADJ_MATRIX_PATH = MODEL_DIR / "adjacency_matrix.npy"
    SCALER_MEAN_PATH = MODEL_DIR / "scaler_mean.npy"
    SCALER_STD_PATH = MODEL_DIR / "scaler_std.npy"
    
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
        'waterlevel', 'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'rain', 'wind_speed_10m', 'surface_pressure', 'wind_gusts_10m',
        'wind_direction_10m', 'cloud_cover'
    ]
    
    # Model hyperparameters
    NUM_STATIONS = 5
    NUM_FEATURES = 10
    TARGET_FEATURE_IDX = 0  # waterlevel
    LOOKBACK_WINDOW = 24 * 3  # 3 days
    FORECAST_HORIZONS = [1, 3, 6, 12, 24, 48]
    
    # STGAE params
    STGAE_GCN_HIDDEN = 64
    STGAE_GRU_HIDDEN_FACTOR = 64
    
    # GAT-Transformer params
    GAT_HIDDEN_DIM = 32
    GAT_HEADS = 4
    TRANSFORMER_D_MODEL = GAT_HIDDEN_DIM * GAT_HEADS
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 3
    TRANSFORMER_FF_DIM = 256
    
    # API endpoints
    WATER_API_BASE = "http://121.58.193.173:8080/water/map_list.do"
    WEATHER_API_BASE = "https://api.open-meteo.com/v1/forecast"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# --- Model Definitions---

class FrozenSTGAEEncoder(nn.Module):
    def __init__(self, in_features, gcn_hidden, gru_hidden, encoder_weights_path):
        super().__init__()
        self.encoder_gcn = GCNConv(in_features, gcn_hidden)
        self.encoder_gru = nn.GRU(gcn_hidden * config.NUM_STATIONS, gru_hidden, batch_first=True)
        self.activation = nn.Tanh()
        
        # Load pre-trained weights
        encoder_state = torch.load(encoder_weights_path, map_location=config.DEVICE)
        self.encoder_gcn.load_state_dict(encoder_state['encoder_gcn'])
        self.encoder_gru.load_state_dict(encoder_state['encoder_gru'])
        
        for param in self.parameters():
            param.requires_grad = False
    
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
            in_features=config.STGAE_GCN_HIDDEN,
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
    def __init__(self, config):
        super().__init__()
        self.gat = GraphAttentionLayer(
            in_features=config.NUM_FEATURES,
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
        
        x_flat = x.reshape(-1, num_features)
        edge_index_batched = edge_index.clone()
        for i in range(1, batch_size * seq_len):
            edge_index_batched = torch.cat(
                [edge_index_batched, edge_index + i * num_nodes], dim=1
            )
        
        gat_out = self.gat(x_flat, edge_index_batched)
        gat_features = gat_out.reshape(batch_size, seq_len, num_nodes, -1)
        transformer_input = gat_features.mean(dim=2)
        transformer_out = self.transformer(transformer_input)
        last_timestep = transformer_out[:, -1, :]
        predictions = self.forecast_head(last_timestep)
        
        return predictions

# --- STGAE for imputation ---
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
        
        # Encoder
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

        # Decoder
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

# --- Data fetching and processing ---

class DataFetcher:
    def __init__(self):
        self.config = config
        
    def fetch_water_levels(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch water level data from API"""
        all_data = []
        current_time = start_time
        
        while current_time <= end_time:
            ymdhm = current_time.strftime("%Y%m%d%H%M")
            url = f"{self.config.WATER_API_BASE}?ymdhm={ymdhm}"
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        if item['obsnm'] in ['Montalban', 'Nangka', 'San Mateo-1', 'Sto Nino', 'Tumana Bridge']:
                            all_data.append({
                                'datetime': current_time,
                                'station': item['obsnm'],
                                'waterlevel': float(item['wl']) if item['wl'] else np.nan
                            })
                            
            except Exception as e:
                logger.warning(f"Failed to fetch water data for {ymdhm}: {e}")
            
            current_time += timedelta(hours=1)
        
        df = pd.DataFrame(all_data)
        if not df.empty:
            # Map station names
            station_map = {
                'Montalban': 'MONTALBAN',
                'Nangka': 'NANGKA', 
                'San Mateo-1': 'SAN MATEO',
                'Sto Nino': 'STO NINO',
                'Tumana Bridge': 'TUMANA'
            }
            df['station'] = df['station'].map(station_map)
            df = df.pivot(index='datetime', columns='station', values='waterlevel')
        
        return df
    
    def fetch_weather_data(self, start_time: datetime, end_time: datetime, lat: float, lon: float) -> pd.DataFrame:
        """Fetch weather data from Open-Meteo API"""
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': 'temperature_2m,relative_humidity_2m,dew_point_2m,rain,wind_speed_10m,wind_gusts_10m,cloud_cover,wind_direction_10m,surface_pressure',
            'timezone': 'Asia/Singapore',
            'start_date': start_time.strftime('%Y-%m-%d'),
            'end_date': end_time.strftime('%Y-%m-%d')
        }
        
        try:
            response = requests.get(self.config.WEATHER_API_BASE, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['hourly'])
                df['datetime'] = pd.to_datetime(df['time'])
                df = df.set_index('datetime')
                df = df.drop('time', axis=1)
                return df
        except Exception as e:
            logger.warning(f"Failed to fetch weather data: {e}")
        
        return pd.DataFrame()
    
    def prepare_input_data(self) -> np.ndarray:
        """Prepare input data for the model"""
        # 1. Truncate current time to the beginning of the hour for alignment
        now = datetime.now()
        end_time = now.replace(minute=0, second=0, microsecond=0)
        
        # Get data for the last lookback_window hours plus a buffer
        start_time = end_time - timedelta(hours=config.LOOKBACK_WINDOW + 24)
        
        # Fetch water levels
        water_df = self.fetch_water_levels(start_time, end_time)
        
        # Create a clean, hourly index for alignment
        timesteps = pd.date_range(start=start_time, end=end_time, freq='h')
        feature_array = np.full((len(timesteps), config.NUM_STATIONS, config.NUM_FEATURES), np.nan)
        
        # 2. Use .reindex() for robust data alignment, which handles missing hours
        # Fill water levels
        aligned_water_df = water_df.reindex(timesteps)
        for s_idx, station in enumerate(config.STATION_ORDER):
            if station in aligned_water_df.columns:
                water_values = aligned_water_df[station].values
                feature_array[:, s_idx, 0] = water_values
        
        # Fetch and fill weather data for each station
        for s_idx, station in enumerate(config.STATION_ORDER):
            lat, lon = config.STATION_CONFIG[station]['coords']
            weather_df = self.fetch_weather_data(start_time, end_time, lat, lon)
            
            if not weather_df.empty:
                aligned_weather_df = weather_df.reindex(timesteps)
                for f_idx, feature in enumerate(config.FEATURE_ORDER[1:], 1):  # Skip waterlevel
                    if feature in aligned_weather_df.columns:
                        feature_values = aligned_weather_df[feature].values
                        feature_array[:, s_idx, f_idx] = feature_values
        
        # Return only the most recent lookback_window timesteps
        return feature_array[-config.LOOKBACK_WINDOW:]

# --- Model Manager ---

class ModelManager:
    def __init__(self):
        self.models = {}
        self.stgae = None
        self.mean = None
        self.std = None
        self.edge_index = None
        self.load_scalers()
        self.load_adjacency()
        
    def load_scalers(self):
        """Load data scalers"""
        try:
            self.mean = np.load(config.SCALER_MEAN_PATH)
            self.std = np.load(config.SCALER_STD_PATH)
        except Exception as e:
            logger.error(f"Failed to load scalers: {e}")
            self.mean = np.zeros(config.NUM_FEATURES)
            self.std = np.ones(config.NUM_FEATURES)
    
    def load_adjacency(self):
        """Load adjacency matrix"""
        try:
            adj_matrix = np.load(config.ADJ_MATRIX_PATH)
            self.edge_index = torch.tensor(
                np.array(np.where(adj_matrix > 0)), 
                dtype=torch.long
            ).to(config.DEVICE)
        except Exception as e:
            logger.error(f"Failed to load adjacency matrix: {e}")
            # Create fully connected graph as fallback
            adj_matrix = np.ones((config.NUM_STATIONS, config.NUM_STATIONS)) - np.eye(config.NUM_STATIONS)
            self.edge_index = torch.tensor(
                np.array(np.where(adj_matrix > 0)), 
                dtype=torch.long
            ).to(config.DEVICE)
    
    def load_model(self, model_type: ModelType):
        """Load and cache a model"""
        if model_type in self.models:
            return self.models[model_type]
        
        try:
            if model_type == ModelType.FULL:
                model = STGAEGATTransformer(config, config.STGAE_ENCODER_PATH)
                model.load_state_dict(torch.load(config.FULL_MODEL_PATH, map_location=config.DEVICE))
            else:  # ABLATED
                model = AblatedGATTransformer(config)
                model.load_state_dict(torch.load(config.ABLATED_MODEL_PATH, map_location=config.DEVICE))
            
            model.to(config.DEVICE)
            model.eval()
            self.models[model_type] = model
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    def load_stgae(self):
        """Load STGAE for imputation"""
        if self.stgae is not None:
            return self.stgae
        
        try:
            self.stgae = STGAE(
                in_features=config.NUM_FEATURES,
                gcn_hidden=config.STGAE_GCN_HIDDEN,
                gru_hidden=config.STGAE_GRU_HIDDEN_FACTOR * config.NUM_STATIONS
            ).to(config.DEVICE)
            
            self.stgae.load_state_dict(torch.load(config.STGAE_FULL_PATH, map_location=config.DEVICE))
            self.stgae.eval()
            return self.stgae
            
        except Exception as e:
            logger.error(f"Failed to load STGAE: {e}")
            return None
    
    def impute_missing_data(self, data: np.ndarray) -> np.ndarray:
        """Impute missing values using STGAE"""
        stgae = self.load_stgae()
        if stgae is None:
            # Fallback: simple forward fill
            df = pd.DataFrame(data.reshape(data.shape[0], -1))
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            return df.values.reshape(data.shape)
        
        # Scale data
        scaled_data = (data - self.mean) / (self.std + 1e-8)
        
        # Process through STGAE
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
            input_tensor = torch.nan_to_num(input_tensor)
            reconstruction = stgae(input_tensor, self.edge_index).squeeze(0).cpu().numpy()
        
        # Only fill NaN values
        is_nan = np.isnan(scaled_data)
        scaled_data[is_nan] = reconstruction[is_nan]
        
        # Inverse scale
        imputed_data = (scaled_data * (self.std + 1e-8)) + self.mean
        return imputed_data
    
    def predict(self, input_data: np.ndarray, model_type: ModelType) -> Dict:
        """Make predictions using the specified model"""
        # Impute missing values
        input_data = self.impute_missing_data(input_data)
        
        # Scale data
        scaled_data = (input_data - self.mean) / (self.std + 1e-8)
        
        # Prepare tensor
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        input_tensor = torch.nan_to_num(input_tensor)
        
        # Get model and predict
        model = self.load_model(model_type)
        
        with torch.no_grad():
            predictions = model(input_tensor, self.edge_index)
        
        # Process predictions
        results = {}
        for horizon in config.FORECAST_HORIZONS:
            pred_scaled = predictions[horizon].cpu().numpy()[0]  # Shape: (num_stations,)
            # Inverse scale (only for water level feature)
            pred_original = (pred_scaled * (self.std[0] + 1e-8)) + self.mean[0]
            results[horizon] = pred_original.tolist()
        
        return results

# --- FastAPI Application ---

app = FastAPI(
    title="River Level Prediction API",
    description="Predicts river water levels using STGAE-GAT-Transformer models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
data_fetcher = DataFetcher()
model_manager = ModelManager()

# --- API Models ---

class PredictionRequest(BaseModel):
    model_type: ModelType = ModelType.FULL
    include_current_levels: bool = True

class StationPrediction(BaseModel):
    station: str
    current_level: Optional[float]
    predictions: Dict[int, float]

class PredictionResponse(BaseModel):
    model_type: str
    timestamp: str
    stations: List[StationPrediction]
    
# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "message": "River Level Prediction API",
        "endpoints": {
            "/predict": "Get predictions for all stations",
            "/health": "Check API health",
            "/stations": "Get list of stations",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": config.DEVICE,
        "models_available": {
            "full": config.FULL_MODEL_PATH.exists(),
            "ablated": config.ABLATED_MODEL_PATH.exists()
        }
    }

@app.get("/stations")
async def get_stations():
    return {
        "stations": config.STATION_ORDER,
        "station_details": config.STATION_CONFIG
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Get water level predictions for all stations
    
    - **model_type**: Choose between 'full' or 'ablated' model
    - **include_current_levels**: Include current water levels in response
    """
    try:
        # Fetch and prepare input data
        logger.info(f"Fetching input data for {request.model_type} model")
        input_data = data_fetcher.prepare_input_data()
        
        # Get current levels if requested
        current_levels = {}
        if request.include_current_levels:
            # Get the last timestep's water levels
            for s_idx, station in enumerate(config.STATION_ORDER):
                current_level = input_data[-1, s_idx, 0]
                current_levels[station] = float(current_level) if not np.isnan(current_level) else None
        
        # Make predictions
        logger.info("Making predictions")
        predictions = model_manager.predict(input_data, request.model_type)
        
        # Format response
        stations = []
        for s_idx, station in enumerate(config.STATION_ORDER):
            station_pred = StationPrediction(
                station=station,
                current_level=current_levels.get(station),
                predictions={
                    horizon: predictions[horizon][s_idx] 
                    for horizon in config.FORECAST_HORIZONS
                }
            )
            stations.append(station_pred)
        
        return PredictionResponse(
            model_type=request.model_type,
            timestamp=datetime.now().isoformat(),
            stations=stations
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/station/{station_name}")
async def predict_station(
    station_name: str, 
    model_type: ModelType = Query(ModelType.FULL)
):
    """Get predictions for a specific station"""
    if station_name not in config.STATION_ORDER:
        raise HTTPException(status_code=404, detail=f"Station {station_name} not found")
    
    try:
        input_data = data_fetcher.prepare_input_data()
        predictions = model_manager.predict(input_data, model_type)
        
        station_idx = config.STATION_ORDER.index(station_name)
        
        return {
            "station": station_name,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "predictions": {
                horizon: predictions[horizon][station_idx]
                for horizon in config.FORECAST_HORIZONS
            }
        }
        
    except Exception as e:
        logger.error(f"Station prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)