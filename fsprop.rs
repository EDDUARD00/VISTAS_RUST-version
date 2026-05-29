use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;
use crate::mathhelps::{trapz2d, interp1};
use std::time::Instant;

/// Structure to hold the Far-Field Propagation results
pub struct FsPropResult {
    pub uxyffc: Array3<f64>,
    pub uxyffs: Array3<f64>,
    pub rff: f64,
}

/// Helper: Performs an in-place 2D Fast Fourier Transform
fn fft2d(grid: &mut Array2<Complex64>) {
    let (rows, cols) = (grid.nrows(), grid.ncols());
    let mut planner = FftPlanner::new();

    // FFT over rows
    let fft_row = planner.plan_fft_forward(cols);
    for i in 0..rows {
        let mut row_buf = grid.row(i).to_vec();
        fft_row.process(&mut row_buf);
        for j in 0..cols { grid[[i, j]] = row_buf[j]; }
    }

    // FFT over columns
    let fft_col = planner.plan_fft_forward(rows);
    for j in 0..cols {
        let mut col_buf = grid.column(j).to_vec();
        fft_col.process(&mut col_buf);
        for i in 0..rows { grid[[i, j]] = col_buf[i]; }
    }
}

/// Helper: Performs a 2D FFT Shift (swaps quadrants)
fn fftshift2d(input: &Array2<Complex64>) -> Array2<Complex64> {
    let (rows, cols) = (input.nrows(), input.ncols());
    let mut output = Array2::zeros((rows, cols));
    let mid_r = (rows + 1) / 2;
    let mid_c = (cols + 1) / 2;

    for i in 0..rows {
        let shift_i = (i + mid_r) % rows;
        for j in 0..cols {
            let shift_j = (j + mid_c) % cols;
            output[[shift_i, shift_j]] = input[[i, j]];
        }
    }
    output
}

pub fn fsprop(
    ur: &Array2<f64>,
    lambda: f64,
    d: f64,
    cr: usize,
    cxy: usize,
    r_cavity: f64,
    dr: f64,
    lvec: &Array1<i64>
) -> FsPropResult {
    let t_start = Instant::now();
    let nm = ur.nrows();

    // 1. Grid formatting
    let r_ext = r_cavity + (cxy as f64 - 2.0 * cr as f64 - 1.0) * dr / 2.0;
    let rff = lambda * d * (cxy as f64) / (4.0 * r_ext); // 4.0 captures the /2/2 in MATLAB
    
    // Original polar radial vector
    let r_vec = Array1::linspace(0.0, r_cavity, cr + 1);

    // Near-field grid
    let x = Array1::linspace(-r_ext, r_ext, cxy);
    let y = Array1::linspace(r_ext, -r_ext, cxy); 

    // Far-field grid
    let xff = Array1::linspace(-rff, rff, cxy);
    let yff = Array1::linspace(rff, -rff, cxy);
    // Ascending yff for accurate trapz integration
    let yff_asc = Array1::linspace(-rff, rff, cxy); 

    let mut uxyffc_out = Array3::<f64>::zeros((nm, cxy, cxy));
    let mut uxyffs_out = Array3::<f64>::zeros((nm, cxy, cxy));

    // 2. Setup static Phase & Pre-factors
    let mut qt = Array2::<Complex64>::zeros((cxy, cxy));
    let mut h0 = Array2::<Complex64>::zeros((cxy, cxy));
    
    let j_comp = Complex64::new(0.0, 1.0);
    let h0_prefactor = (1.0 / (j_comp * lambda * d)) * (j_comp * 2.0 * PI / lambda * d).exp();

    for i in 0..cxy {
        for j in 0..cxy {
            let nf_rad2 = x[j].powi(2) + y[i].powi(2);
            let ff_rad2 = xff[j].powi(2) + yff[i].powi(2);
            
            qt[[i, j]] = (j_comp * PI / (lambda * d) * nf_rad2).exp();
            h0[[i, j]] = h0_prefactor * (j_comp * PI / (lambda * d) * ff_rad2).exp();
        }
    }

    // 3. Main Spatial Processing Loop
    for k in 0..nm {
        let ur_k = ur.row(k).to_owned();
        let mut uxyc_comp = Array2::<Complex64>::zeros((cxy, cxy));
        let mut uxys_comp = Array2::<Complex64>::zeros((cxy, cxy));

        // Evaluate the polar mode mapping onto Cartesian
        for i in 0..cxy {
            for j in 0..cxy {
                let radius = (x[j].powi(2) + y[i].powi(2)).sqrt();
                let phi = y[i].atan2(x[j]);
                
                let z_val = interp1(&r_vec, &ur_k, radius);
                let l_order = lvec[k] as f64;
                
                let val_c = z_val * (l_order * phi).cos();
                let val_s = z_val * (l_order * phi).sin();
                
                uxyc_comp[[i, j]] = Complex64::new(val_c, 0.0) * qt[[i, j]];
                uxys_comp[[i, j]] = Complex64::new(val_s, 0.0) * qt[[i, j]];
            }
        }

        // Apply 2D FFT
        fft2d(&mut uxyc_comp);
        fft2d(&mut uxys_comp);

        // Apply Shift and Far-Field Normalization
        let shift_c = fftshift2d(&uxyc_comp);
        let shift_s = fftshift2d(&uxys_comp);

        let mut abs2_c = Array2::<f64>::zeros((cxy, cxy));
        let mut abs2_s = Array2::<f64>::zeros((cxy, cxy));

        for i in 0..cxy {
            for j in 0..cxy {
                abs2_c[[i, j]] = (shift_c[[i, j]] * h0[[i, j]]).norm_sqr();
                abs2_s[[i, j]] = (shift_s[[i, j]] * h0[[i, j]]).norm_sqr();
            }
        }

        let integral_c = trapz2d(&xff, &yff_asc, &abs2_c);
        let norm_factor_c = 4.0 * rff.powi(2) / integral_c;
        abs2_c.mapv_inplace(|v| v * norm_factor_c);
        
        if lvec[k] != 0 {
            let integral_s = trapz2d(&xff, &yff_asc, &abs2_s);
            let norm_factor_s = 4.0 * rff.powi(2) / integral_s;
            abs2_s.mapv_inplace(|v| v * norm_factor_s);
        }

        // Assign to Output Matrices
        for i in 0..cxy {
            for j in 0..cxy {
                uxyffc_out[[k, i, j]] = abs2_c[[i, j]];
                uxyffs_out[[k, i, j]] = abs2_s[[i, j]];
            }
        }
    }

    println!("Far-field propagation calc = {:.2?}s", t_start.elapsed().as_secs_f64());

    FsPropResult {
        uxyffc: uxyffc_out,
        uxyffs: uxyffs_out,
        rff,
    }
}