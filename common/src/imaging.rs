use crate::Viewport;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeasurementUnit {
    #[default]
    Um,
    Mm,
    Inches,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StainNormalization {
    #[default]
    None,
    Macenko,
    Vahadane,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RgbaImageData {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

pub fn crop_transparent_edges(image: RgbaImageData) -> RgbaImageData {
    if image.width == 0 || image.height == 0 {
        return image;
    }

    let stride = image.width * 4;
    let mut min_x = image.width;
    let mut min_y = image.height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut found_opaque = false;

    for y in 0..image.height {
        let row_start = y * stride;
        for x in 0..image.width {
            let alpha = image.pixels[row_start + x * 4 + 3];
            if alpha != 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                found_opaque = true;
            }
        }
    }

    if !found_opaque {
        return image;
    }

    let cropped_width = max_x - min_x + 1;
    let cropped_height = max_y - min_y + 1;
    if cropped_width == image.width && cropped_height == image.height {
        return image;
    }

    let mut cropped = Vec::with_capacity(cropped_width * cropped_height * 4);
    for y in min_y..=max_y {
        let row_start = y * stride + min_x * 4;
        let row_end = row_start + cropped_width * 4;
        cropped.extend_from_slice(&image.pixels[row_start..row_end]);
    }

    RgbaImageData {
        width: cropped_width,
        height: cropped_height,
        pixels: cropped,
    }
}

pub fn crop_image_to_viewport_bounds(image: RgbaImageData, viewport: &Viewport) -> RgbaImageData {
    if image.width == 0 || image.height == 0 || viewport.width <= 0.0 || viewport.height <= 0.0 {
        return image;
    }

    let top_left = viewport.image_to_screen(0.0, 0.0);
    let bottom_right = viewport.image_to_screen(viewport.image_width, viewport.image_height);

    let visible_left = top_left.x.min(bottom_right.x).clamp(0.0, viewport.width);
    let visible_top = top_left.y.min(bottom_right.y).clamp(0.0, viewport.height);
    let visible_right = top_left.x.max(bottom_right.x).clamp(0.0, viewport.width);
    let visible_bottom = top_left.y.max(bottom_right.y).clamp(0.0, viewport.height);

    if visible_right <= visible_left || visible_bottom <= visible_top {
        return image;
    }

    let scale_x = image.width as f64 / viewport.width.max(1.0);
    let scale_y = image.height as f64 / viewport.height.max(1.0);

    let crop_left = (visible_left * scale_x)
        .floor()
        .clamp(0.0, image.width as f64) as usize;
    let crop_top = (visible_top * scale_y)
        .floor()
        .clamp(0.0, image.height as f64) as usize;
    let crop_right = (visible_right * scale_x)
        .ceil()
        .clamp(0.0, image.width as f64) as usize;
    let crop_bottom = (visible_bottom * scale_y)
        .ceil()
        .clamp(0.0, image.height as f64) as usize;

    if crop_left == 0 && crop_top == 0 && crop_right == image.width && crop_bottom == image.height {
        return image;
    }

    if crop_right <= crop_left || crop_bottom <= crop_top {
        return image;
    }

    let cropped_width = crop_right - crop_left;
    let cropped_height = crop_bottom - crop_top;
    let stride = image.width * 4;
    let mut cropped = Vec::with_capacity(cropped_width * cropped_height * 4);
    for y in crop_top..crop_bottom {
        let row_start = y * stride + crop_left * 4;
        let row_end = row_start + cropped_width * 4;
        cropped.extend_from_slice(&image.pixels[row_start..row_end]);
    }

    RgbaImageData {
        width: cropped_width,
        height: cropped_height,
        pixels: cropped,
    }
}
