//! Common crate for eov WSI viewer
//!
//! This crate provides high-performance WSI (Whole Slide Image) file handling,
//! tiling, caching, and rendering utilities.

pub mod cache;
pub mod error;
pub mod tile;
pub mod viewport;
pub mod wsi;

pub use cache::TileCache;
pub use error::{Error, Result};
pub use tile::{TileCoord, TileData, TileManager};
pub use viewport::{Viewport, ViewportState};
pub use wsi::{WsiFile, WsiLevel, WsiProperties};
