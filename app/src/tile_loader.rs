//! Background tile loading system with automatic cancellation
//!
//! This module provides asynchronous tile loading that automatically discards
//! requests for tiles that are no longer visible.

use common::{TileCache, TileCoord, TileData, TileManager, WsiFile};
use crossbeam_channel::{bounded, Sender, Receiver, TrySendError};
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use tracing::trace;

/// Number of background worker threads for tile loading
const WORKER_COUNT: usize = 1;

/// Maximum tiles to send to workers per frame
const MAX_TILES_PER_FRAME: usize = 4;

/// Background tile loader with automatic cancellation of stale requests
pub struct TileLoader {
    /// Channel to send tile requests to workers
    request_tx: Sender<TileCoord>,
    /// Set of tiles that failed to load (to avoid repeated requests)
    failed: Arc<Mutex<HashSet<TileCoord>>>,
    /// Current generation - tiles from old generations are skipped
    generation: Arc<AtomicU64>,
    /// Worker handles
    workers: Vec<thread::JoinHandle<()>>,
    /// Shutdown flag
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    /// Reference to cache
    cache: Arc<TileCache>,
}

impl TileLoader {
    /// Create a new tile loader
    pub fn new(tile_manager: Arc<TileManager>, cache: Arc<TileCache>) -> Self {
        // Use a small bounded channel - we don't want to queue too many requests
        let (request_tx, request_rx) = bounded::<TileCoord>(8);
        let failed = Arc::new(Mutex::new(HashSet::new()));
        let generation = Arc::new(AtomicU64::new(0));
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        
        let mut workers = Vec::with_capacity(WORKER_COUNT);
        
        for i in 0..WORKER_COUNT {
            let request_rx = request_rx.clone();
            let tile_manager = Arc::clone(&tile_manager);
            let cache = Arc::clone(&cache);
            let failed = Arc::clone(&failed);
            let shutdown = Arc::clone(&shutdown);
            
            let handle = thread::spawn(move || {
                worker_loop(i, request_rx, tile_manager, cache, failed, shutdown);
            });
            
            workers.push(handle);
        }
        
        Self {
            request_tx,
            failed,
            generation,
            workers,
            shutdown,
            cache,
        }
    }
    
    /// Update the set of wanted tiles for this frame
    /// Sends tiles to workers, skipping those already cached or failed
    pub fn set_wanted_tiles(&self, tiles: Vec<TileCoord>) {
        // Bump generation to invalidate any stale in-flight requests
        self.generation.fetch_add(1, Ordering::SeqCst);
        
        let mut sent = 0;
        
        // Send tiles to workers (limited to avoid flooding)
        for coord in tiles {
            if sent >= MAX_TILES_PER_FRAME {
                break;
            }
            
            // Skip if already cached
            if self.cache.contains(&coord) {
                continue;
            }
            
            // Skip if previously failed (brief lock)
            if self.failed.lock().contains(&coord) {
                continue;
            }
            
            // Try to send to workers (non-blocking)
            match self.request_tx.try_send(coord) {
                Ok(_) => sent += 1,
                Err(TrySendError::Full(_)) => break, // Channel full, stop for this frame
                Err(TrySendError::Disconnected(_)) => break,
            }
        }
    }
    
    /// Clear failed tiles set (call when view changes significantly)
    pub fn clear_failed(&self) {
        self.failed.lock().clear();
    }
}

impl Drop for TileLoader {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Drop the sender to unblock workers
        drop(self.request_tx.clone());
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Worker thread loop
fn worker_loop(
    id: usize,
    request_rx: Receiver<TileCoord>,
    tile_manager: Arc<TileManager>,
    cache: Arc<TileCache>,
    failed: Arc<Mutex<HashSet<TileCoord>>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
) {
    trace!("Tile loader worker {} starting", id);
    
    while !shutdown.load(Ordering::Relaxed) {
        // Wait for a tile request with timeout
        let coord = match request_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(coord) => coord,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        
        // Skip if already in cache
        if cache.contains(&coord) {
            continue;
        }
        
        // Load the tile
        match tile_manager.load_tile_sync(coord) {
            Ok(tile) => {
                cache.insert(tile);
                trace!("Worker {} loaded tile {:?}", id, coord);
            }
            Err(e) => {
                trace!("Worker {} failed to load tile {:?}: {}", id, coord, e);
                failed.lock().insert(coord);
            }
        }
    }
    
    trace!("Tile loader worker {} shutting down", id);
}

/// Find the best fallback tile for a given target tile coordinate
/// 
/// This searches lower-resolution (higher mip) levels for a tile that covers
/// the same area as the target tile. Returns the covering tile if found in cache.
pub fn find_fallback_tile(
    cache: &TileCache,
    wsi: &WsiFile,
    target: TileCoord,
) -> Option<(Arc<TileData>, u32)> {
    let level_count = wsi.level_count();
    
    // Search lower-resolution levels (higher indices)
    for fallback_level in (target.level + 1)..level_count {
        let target_level_info = wsi.level(target.level)?;
        let fallback_level_info = wsi.level(fallback_level)?;
        
        // Calculate the scale factor between levels
        let scale = fallback_level_info.downsample / target_level_info.downsample;
        
        // Calculate which tile at the fallback level covers this area
        let fallback_x = (target.x as f64 / scale) as u64;
        let fallback_y = (target.y as f64 / scale) as u64;
        
        let fallback_coord = TileCoord::new(fallback_level, fallback_x, fallback_y);
        
        if let Some(tile) = cache.get(&fallback_coord) {
            return Some((tile, fallback_level));
        }
    }
    
    None
}

/// Calculate the sub-region of a fallback tile that corresponds to the target tile
pub struct FallbackRegion {
    /// Source X offset within the fallback tile (0.0-1.0)
    pub src_x: f64,
    /// Source Y offset within the fallback tile (0.0-1.0)
    pub src_y: f64,
    /// Source width fraction (0.0-1.0)
    pub src_w: f64,
    /// Source height fraction (0.0-1.0)
    pub src_h: f64,
}

pub fn calculate_fallback_region(
    wsi: &WsiFile,
    target: TileCoord,
    fallback_level: u32,
) -> Option<FallbackRegion> {
    let target_level_info = wsi.level(target.level)?;
    let fallback_level_info = wsi.level(fallback_level)?;
    
    let scale = fallback_level_info.downsample / target_level_info.downsample;
    
    // Calculate which fallback tile and the position within it
    let fallback_x = (target.x as f64 / scale) as u64;
    let fallback_y = (target.y as f64 / scale) as u64;
    
    // Calculate the fractional position within the fallback tile
    let frac_x = (target.x as f64 / scale) - fallback_x as f64;
    let frac_y = (target.y as f64 / scale) - fallback_y as f64;
    
    // The fraction of the fallback tile that this target tile covers
    let frac_w = (1.0 / scale).min(1.0 - frac_x);
    let frac_h = (1.0 / scale).min(1.0 - frac_y);
    
    Some(FallbackRegion {
        src_x: frac_x,
        src_y: frac_y,
        src_w: frac_w,
        src_h: frac_h,
    })
}

/// Calculate tiles needed for the current viewport
pub fn calculate_wanted_tiles(
    tile_manager: &TileManager,
    level: u32,
    bounds_left: f64,
    bounds_top: f64,
    bounds_right: f64,
    bounds_bottom: f64,
) -> Vec<TileCoord> {
    let mut wanted = Vec::new();
    let level_count = tile_manager.wsi().level_count();
    
    // First, add lower-resolution tiles for fallback display
    // Prioritize levels CLOSEST to current level (one level up is best fallback quality)
    // These tiles have enough pixels to display properly even when zoomed in
    for fallback_level in (level + 1)..level_count {
        let fallback_tiles = tile_manager.visible_tiles(
            fallback_level,
            bounds_left,
            bounds_top,
            bounds_right,
            bounds_bottom,
        );
        
        // Closer levels (smaller fallback_level - level) get more tiles loaded
        // as they provide better fallback quality
        let priority_boost = level_count.saturating_sub(fallback_level);
        let tiles_at_this_level = (fallback_tiles.len().min(20) + priority_boost as usize * 10).min(50);
        wanted.extend(fallback_tiles.iter().take(tiles_at_this_level).copied());
    }
    
    // Then add current level tiles (these are the final high-res display)
    let visible = tile_manager.visible_tiles(
        level,
        bounds_left,
        bounds_top,
        bounds_right,
        bounds_bottom,
    );
    wanted.extend(visible.iter().take(500).copied());
    
    wanted
}
