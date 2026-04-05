//! Error types for the common crate

use thiserror::Error;

/// Result type alias using our Error type
pub type Result<T> = std::result::Result<T, Error>;

/// Common crate error types
#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to open WSI file: {0}")]
    OpenFile(String),

    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    #[error("Failed to read tile at level {level}, x={x}, y={y}: {message}")]
    ReadTile {
        level: u32,
        x: u64,
        y: u64,
        message: String,
    },

    #[error("Invalid level: {0} (max: {1})")]
    InvalidLevel(u32, u32),

    #[error("Invalid coordinates: ({x}, {y}) at level {level}")]
    InvalidCoordinates { x: i64, y: i64, level: u32 },

    #[error("OpenSlide error: {0}")]
    OpenSlide(String),

    #[error("Image encoding error: {0}")]
    ImageEncode(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("File not found: {0}")]
    FileNotFound(String),
}
