//! Stable ABI types for the plugin dynamic library interface.
//!
//! Uses [`abi_stable`] to define types that can safely cross the Rust dynamic
//! library boundary. The plugin exports a `#[no_mangle]` function that returns
//! a [`PluginVTable`], and the host loads it with
//! [`abi_stable::library::RawLibrary`].

use abi_stable::std_types::{RString, RVec};
use abi_stable::StableAbi;

/// FFI-safe toolbar button registration data.
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ToolbarButtonFFI {
    pub button_id: RString,
    pub tooltip: RString,
    pub icon_svg: RString,
    pub action_id: RString,
}

/// FFI-safe response from a plugin action handler.
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ActionResponseFFI {
    /// If `true`, the host should open this plugin's `.slint` UI window.
    pub open_window: bool,
}

/// VTable of function pointers exported by each plugin shared library.
///
/// Each plugin crate exports:
/// ```ignore
/// #[unsafe(no_mangle)]
/// pub extern "C" fn eov_get_plugin_vtable() -> PluginVTable { ... }
/// ```
#[repr(C)]
#[derive(StableAbi, Copy, Clone)]
pub struct PluginVTable {
    /// Returns the toolbar buttons this plugin wants to register.
    pub get_toolbar_buttons: extern "C" fn() -> RVec<ToolbarButtonFFI>,

    /// Called when a toolbar button registered by this plugin is clicked.
    pub on_action: extern "C" fn(action_id: RString) -> ActionResponseFFI,

    /// Called when a callback defined in the plugin's `.slint` UI is invoked.
    /// `callback_name` is the kebab-case name of the callback as declared in
    /// the `.slint` file.
    pub on_ui_callback: extern "C" fn(callback_name: RString),

    /// Returns viewport filter descriptors this plugin provides.
    /// May return an empty vec if the plugin has no viewport filters.
    pub get_viewport_filters: extern "C" fn() -> RVec<ViewportFilterFFI>,

    /// Apply a CPU viewport filter in-place.
    /// `filter_id` identifies which filter to apply.
    /// `rgba_data` is a mutable pointer to width*height*4 bytes of RGBA8 data.
    /// Returns `true` if the filter was applied successfully.
    pub apply_filter_cpu:
        extern "C" fn(filter_id: RString, rgba_data: *mut u8, len: u32, width: u32, height: u32) -> bool,

    /// Apply a GPU viewport filter using raw Vulkan handles.
    /// `filter_id` identifies which filter to apply.
    /// `ctx` is a pointer to a `GpuFilterContextFFI` struct with the Vulkan
    /// device, image, and sync fence. The plugin must signal the fence when done.
    /// Returns `true` if the filter was applied successfully.
    pub apply_filter_gpu: extern "C" fn(filter_id: RString, ctx: *const GpuFilterContextFFI) -> bool,

    /// Enable or disable a viewport filter.
    pub set_filter_enabled: extern "C" fn(filter_id: RString, enabled: bool),
}

/// FFI-safe GPU filter context passed from the host to the plugin.
///
/// Contains raw Vulkan handles for zero-copy GPU memory access. The plugin
/// creates its own Vulkan pipeline on the shared device and submits commands
/// that operate on `vk_image`. The plugin **must** signal `vk_done_fence`
/// when its GPU work is complete.
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct GpuFilterContextFFI {
    /// `VkDevice` handle (as `u64`).
    pub vk_device: u64,
    /// `VkPhysicalDevice` handle (as `u64`).
    pub vk_physical_device: u64,
    /// `VkQueue` handle the plugin may submit to.
    pub vk_queue: u64,
    /// Queue family index for `vk_queue`.
    pub queue_family_index: u32,
    /// `VkImage` handle for the RGBA8 frame.
    pub vk_image: u64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// `VkFormat` value (e.g. `VK_FORMAT_R8G8B8A8_UNORM` = 37).
    pub vk_format: u32,
    /// `VkFence` handle — the plugin **must** signal this when its GPU
    /// commands have finished executing on `vk_image`.
    pub vk_done_fence: u64,
}

/// FFI-safe viewport filter descriptor.
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ViewportFilterFFI {
    pub filter_id: RString,
    pub name: RString,
    pub supports_cpu: bool,
    pub supports_gpu: bool,
    pub enabled: bool,
}

/// The type of the factory function each plugin exports.
pub type GetPluginVTableFn = unsafe extern "C" fn() -> PluginVTable;

/// Symbol name the host looks for in plugin shared libraries.
pub const PLUGIN_VTABLE_SYMBOL: &[u8] = b"eov_get_plugin_vtable\0";

/// Returns the expected shared library filename for a plugin on the current
/// platform, derived from the plugin id.
pub fn plugin_library_filename(id: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("lib{id}.dylib")
    } else if cfg!(target_os = "windows") {
        format!("{id}.dll")
    } else {
        format!("lib{id}.so")
    }
}
