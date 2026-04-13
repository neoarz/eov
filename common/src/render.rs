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
