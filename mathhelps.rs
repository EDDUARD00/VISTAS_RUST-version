use ndarray::{Array1, Array2};

// 1D Trapezoidal Integration
pub fn trapz(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len();
    if n < 2 { return 0.0; }
    let mut sum = 0.0;
    for i in 0..(n - 1) {
        let dx = x[i + 1] - x[i];
        sum += 0.5 * (y[i + 1] + y[i]) * dx;
    }
    sum
}

// 2D Trapezoidal Integration over a Cartesian grid
pub fn trapz2d(x: &Array1<f64>, y: &Array1<f64>, z: &Array2<f64>) -> f64 {
    let rows = z.nrows();
    let mut inner_integrals = Array1::<f64>::zeros(rows);
    
    for i in 0..rows {
        let row_data = z.index_axis(ndarray::Axis(0), i).to_owned();
        inner_integrals[i] = trapz(x, &row_data);
    }
    
    // Integrate the resulting 1D array over the y-axis
    trapz(y, &inner_integrals)
}

// 2D Cross-Correlation (Spatial Domain)
// Computes the cross-correlation of A and B. 
// Assumes B is symmetric (which is true for the circular waveguide mask).
pub fn xcorr2(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (ma, na) = (a.nrows(), a.ncols());
    let (mb, nb) = (b.nrows(), b.ncols());
    let out_rows = ma + mb - 1;
    let out_cols = na + nb - 1;
    let mut out = Array2::<f64>::zeros((out_rows, out_cols));

    for i in 0..out_rows {
        for j in 0..out_cols {
            let mut sum = 0.0;
            for m in 0..mb {
                for n in 0..nb {
                    let row_a = i as isize - m as isize;
                    let col_a = j as isize - n as isize;
                    if row_a >= 0 && row_a < ma as isize && col_a >= 0 && col_a < na as isize {
                        sum += a[[row_a as usize, col_a as usize]] * b[[m, n]];
                    }
                }
            }
            out[[i, j]] = sum;
        }
    }
    out
}


/// 1D Linear Interpolation for fsprop
pub fn interp1(x: &Array1<f64>, y: &Array1<f64>, xq: f64) -> f64 {
    let n = x.len();
    if xq <= x[0] { return y[0]; }
    if xq >= x[n - 1] { return 0.0; } // Modes decay to 0 outside the boundary

    for i in 0..(n - 1) {
        if xq >= x[i] && xq <= x[i + 1] {
            let t = (xq - x[i]) / (x[i + 1] - x[i]);
            return y[i] + t * (y[i + 1] - y[i]);
        }
    }
    0.0
}