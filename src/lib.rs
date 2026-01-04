//! Hinton's Forward-Forward
//!

mod mat;

pub use mat::Mat;

// piecewise linear
pub fn get_lr_scale(epoch: usize, max_epoch: usize) -> f32 {
    if epoch < max_epoch / 2 {
        1.0
    } else {
        (1.0 + 2.0 * (max_epoch - epoch) as f32) / max_epoch as f32
    }
}

pub fn get_cosine_lr_scale(epoch: usize, max_epoch: usize) -> f32 {
    let floor = 0.01;
    let progress = epoch as f32 / max_epoch as f32;
    let cos_out = (progress * std::f32::consts::PI).cos();

    // Smoothly interpolate between 1.0 and floor
    floor + 0.5 * (1.0 - floor) * (1.0 + cos_out)
}
