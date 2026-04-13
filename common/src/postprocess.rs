//! Post-processing functions for CPU-rendered RGBA buffers.
//!
//! Provides gamma/brightness/contrast adjustment via a 256-entry LUT and
//! sharpening via a 3×3 Laplacian unsharp mask. All operations work on
//! raw `[u8]` RGBA buffers with no GUI framework dependency.

use rayon::prelude::*;

/// Apply gamma, brightness, and contrast adjustment to an RGBA buffer.
///
/// The per-channel transform is:
///   `output = clamp(((input / 255)^(1/γ) + brightness - 0.5) × contrast + 0.5, 0, 1) × 255`
///
/// A 256-entry lookup table is built once and applied to R, G, B channels
/// (alpha is left unchanged).
pub fn apply_adjustments(buffer: &mut [u8], gamma: f32, brightness: f32, contrast: f32) {
    let inv_gamma = if gamma > 0.001 { 1.0 / gamma } else { 1.0 };
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let normalized = i as f32 / 255.0;
        let g = normalized.powf(inv_gamma);
        let b = g + brightness;
        let c = (b - 0.5) * contrast + 0.5;
        *entry = (c * 255.0).clamp(0.0, 255.0) as u8;
    }
    buffer.par_chunks_mut(4).for_each(|chunk| {
        chunk[0] = lut[chunk[0] as usize];
        chunk[1] = lut[chunk[1] as usize];
        chunk[2] = lut[chunk[2] as usize];
    });
}

/// Apply 3×3 Laplacian unsharp mask sharpening to an RGBA buffer.
///
/// For each interior pixel, the detail signal is computed as:
///   `detail = 4 × center − (N + S + W + E)`
/// and added back with the given `sharpness` weight:
///   `output = clamp(center + sharpness × detail, 0, 255)`
///
/// Only R, G, B channels are sharpened; alpha is preserved.
/// Edge pixels (first/last row and column) are left unchanged.
pub fn apply_sharpening(buffer: &mut [u8], width: u32, height: u32, sharpness: f32) {
    let w = width as usize;
    let h = height as usize;
    if w < 3 || h < 3 {
        return;
    }
    let src = buffer.to_vec();
    let stride = w * 4;

    buffer
        .par_chunks_mut(stride)
        .enumerate()
        .skip(1)
        .take(h.saturating_sub(2))
        .for_each(|(y, row)| {
            for x in 1..w - 1 {
                let idx = y * stride + x * 4;
                let row_idx = x * 4;
                let n = idx - stride;
                let s = idx + stride;
                let we = idx - 4;
                let e = idx + 4;
                for c in 0..3 {
                    let center = src[idx + c] as f32;
                    let neighbors = src[n + c] as f32
                        + src[s + c] as f32
                        + src[we + c] as f32
                        + src[e + c] as f32;
                    let detail = center * 4.0 - neighbors;
                    let sharpened = center + sharpness * detail;
                    row[row_idx + c] = sharpened.clamp(0.0, 255.0) as u8;
                }
            }
        });
}
