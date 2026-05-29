use ndarray::Array1;
use num_complex::Complex64;
use complex_bessel::{besselj, besselk};

// Computes the eigenvalue equation for the calculation of the LP-modes 
// of a weakly guided step-index waveguide.
//
// * `x` - Radial vector (roots search space)
// * `l` - Azimuthal order of the mode
// * `v` - V-number (normalized frequency)
pub fn fmodes(x: &Array1<f64>, l: i32, v: f64) -> Array1<f64> {
    let mut result = Array1::<f64>::zeros(x.len());
    let l_f64 = l as f64;
    let lp1_f64 = (l + 1) as f64;
    
    for i in 0..x.len() {
        let xi = x[i];
        
        // Mathematically, for guided modes, x must be <= v. 
        // This prevents NaN crashes if the root-finder overshoots.
        let yi = if v >= xi {
            (v.powi(2) - xi.powi(2)).sqrt()
        } else {
            1e-10 // Near-zero boundary fallback
        };
        
        // The complex-bessel crate requires Complex64 inputs
        let cx = Complex64::new(xi, 0.0);
        let cy = Complex64::new(yi, 0.0);
        
        // Calculate Bessel functions safely (extracting the real part)
        // We use a tiny non-zero fallback for denominators to prevent divide-by-zero panics
        let j_lp1 = besselj(lp1_f64, cx).unwrap_or(Complex64::new(0.0, 0.0)).re;
        let j_l   = besselj(l_f64, cx).unwrap_or(Complex64::new(1e-15, 0.0)).re; 
        
        let k_lp1 = besselk(lp1_f64, cy).unwrap_or(Complex64::new(0.0, 0.0)).re;
        let k_l   = besselk(l_f64, cy).unwrap_or(Complex64::new(1e-15, 0.0)).re;
        
        // Evaluate the characteristic equation:
        // x * J_(l+1)(x)/J_l(x) - y * K_(l+1)(y)/K_l(y)
        result[i] = xi * (j_lp1 / j_l) - yi * (k_lp1 / k_l);
    }
    
    result
}