//! Application state management

use common::{TileManager, ViewportState, WsiFile};
use std::path::PathBuf;

/// A single open file with its associated state
pub struct OpenFile {
    /// Unique identifier
    pub id: i32,
    /// File path
    pub path: PathBuf,
    /// Display filename
    pub filename: String,
    /// WSI file handle
    pub wsi: WsiFile,
    /// Tile manager for this file
    pub tile_manager: TileManager,
    /// Viewport state
    pub viewport: ViewportState,
    /// Thumbnail for minimap (RGBA data)
    pub thumbnail: Option<Vec<u8>>,
}

/// Application state
pub struct AppState {
    /// Currently open files
    pub open_files: Vec<OpenFile>,
    /// Active file ID
    pub active_file_id: Option<i32>,
    /// Next file ID
    next_id: i32,
}

impl AppState {
    /// Create a new application state
    pub fn new() -> Self {
        Self {
            open_files: Vec::new(),
            active_file_id: None,
            next_id: 1,
        }
    }

    /// Add a new open file
    pub fn add_file(
        &mut self,
        path: PathBuf,
        wsi: WsiFile,
        tile_manager: TileManager,
        viewport: ViewportState,
        thumbnail: Option<Vec<u8>>,
    ) -> i32 {
        let id = self.next_id;
        self.next_id += 1;

        let filename = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        self.open_files.push(OpenFile {
            id,
            path,
            filename,
            wsi,
            tile_manager,
            viewport,
            thumbnail,
        });

        self.active_file_id = Some(id);
        id
    }

    /// Activate a file by ID
    pub fn activate_file(&mut self, id: i32) {
        if self.open_files.iter().any(|f| f.id == id) {
            self.active_file_id = Some(id);
        }
    }

    /// Close a file by ID
    pub fn close_file(&mut self, id: i32) {
        let index = self.open_files.iter().position(|f| f.id == id);
        
        if let Some(idx) = index {
            self.open_files.remove(idx);
            
            // Update active file
            if self.active_file_id == Some(id) {
                self.active_file_id = self.open_files.get(idx.saturating_sub(1))
                    .or_else(|| self.open_files.first())
                    .map(|f| f.id);
            }
        }
    }

    /// Get a file by ID
    pub fn get_file(&self, id: i32) -> Option<&OpenFile> {
        self.open_files.iter().find(|f| f.id == id)
    }

    /// Get a mutable file by ID
    pub fn get_file_mut(&mut self, id: i32) -> Option<&mut OpenFile> {
        self.open_files.iter_mut().find(|f| f.id == id)
    }

    /// Get the active file
    pub fn active_file(&self) -> Option<&OpenFile> {
        self.active_file_id.and_then(|id| self.get_file(id))
    }

    /// Get the active file mutably
    pub fn active_file_mut(&mut self) -> Option<&mut OpenFile> {
        self.active_file_id.and_then(|id| self.get_file_mut(id))
    }

    /// Get the active viewport
    pub fn active_viewport(&self) -> Option<&ViewportState> {
        self.active_file().map(|f| &f.viewport)
    }

    /// Get the active viewport mutably
    pub fn active_viewport_mut(&mut self) -> Option<&mut ViewportState> {
        self.active_file_mut().map(|f| &mut f.viewport)
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
