// contains the injrpof hehe

use ndarray::Array1;
use std::f64::consts::PI;
use crate::mathhelps::trapz;

/// Structure to hold the current injection profiles
pub struct InjProf {
    pub fcr: Array1<f64>,
    pub fcphi: Array1<f64>,
}

// Computes the normalized current distribution in the active layer
//
// * `r` - Radial vector
// * `r_cavity` - Cavity radius [cm] (R in MATLAB)
// * `rox` - Oxide aperture radius [cm]
// * `rs` - Radial current spreading coefficient [cm]
// * `dphi` - Azimuthal step size [rad]
pub fn injprof(r: &Array1<f64>, r_cavity: f64, rox: f64, rs: f64, dphi: f64) -> InjProf {
    
    // 1. Generate the azimuthal vector (phi) from -pi to pi
    let phi_steps = ((2.0 * PI / dphi).round() as usize) + 1;
    let phi = Array1::from_shape_fn(phi_steps, |i| -PI + (i as f64) * dphi);
    
    // 2. Generate azimuthal current spreading function (fcphi)
    // In MATLAB: fcphi = ones(size(phi));
    let fcphi = Array1::<f64>::ones(phi.len());
    
    // 3. Generate radial current spreading function (fcr)
    // This evaluates the piecewise function: 
    // A = (r < Rox), B = 1 - A; fcr = A + B * exp(-(r - Rox) / rs)
    let mut fcr = Array1::<f64>::zeros(r.len());
    for i in 0..r.len() {
        if r[i] < rox {
            fcr[i] = 1.0;
        } else {
            fcr[i] = (-(r[i] - rox) / rs).exp();
        }
    }
    
    // 4. Normalization calculations
    let fcr_times_r = &fcr * r;
    let int_r = trapz(r, &fcr_times_r);
    let int_phi = trapz(&phi, &fcphi);
    
    // Normalization factor: pi * R^2 / trapz(r, fcr.*r) / trapz(phi, fcphi)
    let norm_factor = PI * r_cavity.powi(2) / int_r / int_phi;
    
    // Apply the normalization factor to the fcr array directly
    fcr.mapv_inplace(|val| val * norm_factor);

    InjProf {
        fcr,
        fcphi,
    }
}