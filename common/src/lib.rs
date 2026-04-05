//! Common crate for eosmol WSI viewer
//! 
//! This crate provides high-performance WSI (Whole Slide Image) file handling,
//! tiling, caching, and rendering utilities.

pub mod wsi;
pub mod tile;
pub mod cache;
pub mod viewport;
pub mod error;

pub use error::{Error, Result};
pub use wsi::{WsiFile, WsiLevel, WsiProperties};
pub use tile::{TileCoord, TileManager, TileData};
pub use cache::TileCache;
pub use viewport::{Viewport, ViewportState};