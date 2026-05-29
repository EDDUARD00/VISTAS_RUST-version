use ndarray::{Array1, Array2};
use num_complex::Complex64;
use complex_bessel::{besselj, besselk};
use std::f64::consts::PI;
use std::time::Instant;

pub struct LPCalcResult {
    pub nm: usize,
    pub ur: Array2<f64>,
}

/// Evaluates the characteristic equation for step-index fibers
fn eval_fmodes(x: f64, l: i32, v: f64) -> f64 {
    let y = (v.powi(2) - x.powi(2)).max(0.0).sqrt();
    let cx = Complex64::new(x, 0.0);
    let cy = Complex64::new(y, 0.0);
    
    let j_lp1 = besselj((l + 1) as f64, cx).unwrap_or(Complex64::new(0.0, 0.0)).re;
    let j_l   = besselj(l as f64, cx).unwrap_or(Complex64::new(1e-15, 0.0)).re;
    
    let k_lp1 = besselk((l + 1) as f64, cy).unwrap_or(Complex64::new(0.0, 0.0)).re;
    let k_l   = besselk(l as f64, cy).unwrap_or(Complex64::new(1e-15, 0.0)).re;
    
    x * (j_lp1 / j_l) - y * (k_lp1 / k_l)
}

/// Custom Root Finder to replace MATLAB's fsolve
/// Uses a bounded Secant Method
fn solve_fmodes(guess: f64, l: i32, v: f64) -> f64 {
    let mut x = guess;
    let dx = 1e-5;
    
    for _ in 0..100 {
        let f = eval_fmodes(x, l, v);
        if f.abs() < 1e-7 {
            break; // Converged
        }
        let f_plus = eval_fmodes(x + dx, l, v);
        let df = (f_plus - f) / dx;
        
        if df.abs() < 1e-12 { break; } // Prevent divide by zero
        
        x -= f / df;
        
        // Physically, the root must lie between 0 and V
        x = x.clamp(1e-4, v - 1e-4);
    }
    x
}

pub fn lpcalc(lambda: f64, nc: f64, dn: f64, a: f64, r_max: f64, cr: usize) -> LPCalcResult {
    let t_start = Instant::now();
    
    // 1. Calculate the normalized frequency (V-number)
    let v = 2.0 * PI * a / lambda * nc * (2.0 * dn).sqrt();
    
    let vvec = [2.42, 3.86, 3.96, 5.16, 5.55, 6.4, 7.1, 7.15, 7.7];
    let lvec = [0, 1, 2, 0, 3, 1, 4, 2, 0, 5, 3, 6, 1, 4];
    let x0 = [1.0, v / 2.0, 3.8, 3.95, 0.75 * v, 0.75 * v, 0.75 * v, 0.75 * v, 0.75 * v];
    
    // 2. Determine number of guided modes
    let mut nm = 1;
    while nm - 1 < vvec.len() && vvec[nm - 1] < v {
        nm += 1;
    }
    
    if v > vvec[nm - 1] {
        println!("WARNING: modes above limit not calculated! V = {:.3}", v);
    }

    let mut ur = Array2::<f64>::zeros((nm, cr + 1));
    let dr = r_max / (cr as f64);
    
    // 3. Compute the spatial profile for each mode
    for i in 0..nm {
        let l = lvec[i];
        let guess = if i < x0.len() { x0[i] } else { 0.75 * v };
        
        // Solve the eigenvalue equation
        let x_val = solve_fmodes(guess, l, v);
        let y_val = (v.powi(2) - x_val.powi(2)).max(0.0).sqrt(); // Corrected Physics Bug!

        let cx = Complex64::new(x_val, 0.0);
        let cy = Complex64::new(y_val, 0.0);
        let j_norm = besselj(l as f64, cx).unwrap_or(Complex64::new(1e-15, 0.0)).re;
        let k_norm = besselk(l as f64, cy).unwrap_or(Complex64::new(1e-15, 0.0)).re;

        // Traverse the radial coordinates
        for j in 0..=cr {
            let r = (j as f64) * dr;
            
            if r < a {
                // Core
                let j_val = besselj(l as f64, Complex64::new(x_val * r / a, 0.0)).unwrap_or(Complex64::new(0.0, 0.0)).re;
                ur[[i, j]] = j_val / j_norm;
            } else {
                // Cladding
                let k_val = besselk(l as f64, Complex64::new(y_val * r / a, 0.0)).unwrap_or(Complex64::new(0.0, 0.0)).re;
                ur[[i, j]] = k_val / k_norm;
            }
            
            // Handle Origin Singularity explicitly
            if j == 0 {
                if l == 0 {
                    let j0_norm = besselj(0.0, cx).unwrap_or(Complex64::new(1e-15, 0.0)).re;
                    ur[[i, 0]] = 1.0 / j0_norm;
                } else {
                    ur[[i, 0]] = 0.0;
                }
            }
        }
    }
    
    println!("LP Modes calculated in {:.2?}s. Generated {} modes.", t_start.elapsed().as_secs_f64(), nm);

    LPCalcResult { nm, ur }
}