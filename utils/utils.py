"""Mamba Strategy Utility Functions Module

This module provides general utility functions, including:
1. Directory creation
2. OU process smoothing
3. Dynamic hysteresis threshold calculation

Author: Tianrun Liao
Date: 2026-02-19
"""

import os
import numpy as np


def _ensure_directory_exists(directory_path: str) -> None:
    """Ensure directory exists, create if it doesn't exist.
    
    Args:
        directory_path: Directory path.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def ornstein_uhlenbeck_smooth(current_exposure: float, prev_exposure: float, theta: float = 0.3) -> float:
    """Apply Ornstein-Uhlenbeck process smoothing to exposure control.
    
    The OU process provides mean-reverting behavior that limits excessive position changes
    while allowing gradual adaptation to new market conditions.
    
    Formula: smoothed = prev_exposure + theta * (current_exposure - prev_exposure)
    
    Args:
        current_exposure: Raw exposure output from Mamba network.
        prev_exposure: Previous day's smoothed exposure.
        theta: Mean reversion speed parameter (0 < theta <= 1). 
               Higher values = less smoothing, lower values = more smoothing.
               
    Returns:
        Smoothed exposure value.
    """
    return prev_exposure + theta * (current_exposure - prev_exposure)


def calculate_dynamic_hysteresis(volatility_history: list, base_threshold: float = 0.08, window: int = 20) -> float:
    """Calculate dynamic hysteresis threshold based on recent volatility.
    
    High volatility periods require higher thresholds to avoid excessive trading,
    while low volatility periods can use lower thresholds for better responsiveness.
    
    Formula: threshold = base_threshold * (1 + recent_vol / avg_vol)
    
    Args:
        volatility_history: List of historical portfolio volatilities.
        base_threshold: Base hysteresis threshold (default 0.08).
        window: Lookback window for volatility calculation (default 20 days).
        
    Returns:
        Dynamic hysteresis threshold.
    """
    if len(volatility_history) < window:
        return base_threshold
    
    recent_vol = np.mean(volatility_history[-window:])
    avg_vol = np.mean(volatility_history)
    
    # Avoid division by zero and ensure reasonable scaling
    if avg_vol == 0:
        return base_threshold
    
    # Scale threshold based on relative volatility
    vol_ratio = max(0.5, min(2.0, recent_vol / avg_vol))  # Clamp between 0.5x and 2.0x
    return base_threshold * vol_ratio