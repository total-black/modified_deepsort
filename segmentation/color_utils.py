"""
Consistent Color Utilities for Segmentation

This module provides deterministic color assignment for segmentation masks
to ensure consistent colors across frames based on object IDs.
"""

def get_color(object_id: int, palette=None):
    """
    Returns a consistent BGR color for a given object ID.
    
    Args:
        object_id: Unique identifier for the object (detection index or track ID)
        palette: Optional custom color palette
        
    Returns:
        BGR color tuple (B, G, R)
    """
    if palette is None:
        palette = [
            (220, 20, 60), (0, 165, 255), (0, 128, 0), (255, 0, 255),
            (255, 140, 0), (30, 144, 255), (128, 0, 128), (60, 180, 75),
            (255, 20, 147), (0, 255, 127), (255, 69, 0), (138, 43, 226),
            (255, 105, 180), (0, 191, 255), (50, 205, 50), (255, 215, 0)
        ]
    
    # Deterministic: same ID -> same color
    return palette[object_id % len(palette)]


def get_default_palette():
    """Get the default color palette."""
    return [
        (220, 20, 60), (0, 165, 255), (0, 128, 0), (255, 0, 255),
        (255, 140, 0), (30, 144, 255), (128, 0, 128), (60, 180, 75),
        (255, 20, 147), (0, 255, 127), (255, 69, 0), (138, 43, 226),
        (255, 105, 180), (0, 191, 255), (50, 205, 50), (255, 215, 0)
    ]
