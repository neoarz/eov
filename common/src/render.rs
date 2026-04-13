use crate::WsiFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderBackend {
    #[default]
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilteringMode {
    Bilinear,
    #[default]
    Trilinear,
    Lanczos3,
}

// ─── Trilinear mip level selection ───────────────────────────────────────────

/// Conservative negative LOD bias applied only to the trilinear mip path.
/// This offsets a slight tendency to choose overly coarse mip blends while
/// keeping the result stable.
pub const TRILINEAR_LOD_BIAS: f64 = -0.25;

/// Result of trilinear level calculation
#[derive(Debug, Clone, Copy)]
pub struct TrilinearLevels {
    /// The higher resolution (lower index) level
    pub level_fine: u32,
    /// The lower resolution (higher index) level
    pub level_coarse: u32,
    /// Blend factor: 0.0 = use level_fine, 1.0 = use level_coarse
    pub blend: f64,
    /// Continuous LOD before the optional trilinear-only bias is applied.
    pub lod_before_bias: f64,
    /// Continuous LOD after applying the trilinear-only bias and clamping.
    pub lod_after_bias: f64,
}

pub fn single_level_trilinear(level: u32) -> TrilinearLevels {
    let lod = level as f64;
    TrilinearLevels {
        level_fine: level,
        level_coarse: level,
        blend: 0.0,
        lod_before_bias: lod,
        lod_after_bias: lod,
    }
}

fn continuous_trilinear_lod(
    target_downsample: f64,
    fine_downsample: f64,
    coarse_downsample: f64,
    level_fine: u32,
) -> f64 {
    let log_target = target_downsample.max(f64::MIN_POSITIVE).ln();
    let log_fine = fine_downsample.max(f64::MIN_POSITIVE).ln();
    let log_coarse = coarse_downsample.max(f64::MIN_POSITIVE).ln();

    let blend = if (log_coarse - log_fine).abs() < 0.001 {
        0.0
    } else {
        ((log_target - log_fine) / (log_coarse - log_fine)).clamp(0.0, 1.0)
    };

    level_fine as f64 + blend
}

pub fn finalize_trilinear_levels(
    level_count: u32,
    lod_before_bias: f64,
    apply_lod_bias: bool,
) -> TrilinearLevels {
    let max_lod = level_count.saturating_sub(1) as f64;
    let lod_after_bias = if apply_lod_bias {
        (lod_before_bias + TRILINEAR_LOD_BIAS).clamp(0.0, max_lod)
    } else {
        lod_before_bias.clamp(0.0, max_lod)
    };
    let level_fine = lod_after_bias.floor() as u32;
    let level_coarse = lod_after_bias.ceil() as u32;

    TrilinearLevels {
        level_fine,
        level_coarse,
        blend: lod_after_bias - level_fine as f64,
        lod_before_bias,
        lod_after_bias,
    }
}

/// Calculate the two mip levels to blend for trilinear filtering.
///
/// Given the target downsample factor (1.0 / viewport zoom) and the WSI file,
/// returns the fine and coarse levels plus the blend factor.
pub fn calculate_trilinear_levels(
    wsi: &WsiFile,
    target_downsample: f64,
    apply_lod_bias: bool,
) -> TrilinearLevels {
    let level_count = wsi.level_count();

    if level_count <= 1 {
        return single_level_trilinear(0);
    }

    let best_level = wsi.best_level_for_downsample(target_downsample);

    let best_info = match wsi.level(best_level) {
        Some(info) => info,
        None => {
            return single_level_trilinear(0);
        }
    };

    let (level_fine, level_coarse) = if target_downsample >= best_info.downsample {
        if best_level + 1 < level_count {
            (best_level, best_level + 1)
        } else {
            return single_level_trilinear(best_level);
        }
    } else if best_level > 0 {
        (best_level - 1, best_level)
    } else {
        return single_level_trilinear(0);
    };

    let lod_before_bias = match (wsi.level(level_fine), wsi.level(level_coarse)) {
        (Some(fine_info), Some(coarse_info)) => continuous_trilinear_lod(
            target_downsample,
            fine_info.downsample,
            coarse_info.downsample,
            level_fine,
        ),
        _ => {
            return single_level_trilinear(level_fine);
        }
    };

    finalize_trilinear_levels(level_count, lod_before_bias, apply_lod_bias)
}
