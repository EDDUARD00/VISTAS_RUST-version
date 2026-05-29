// This contains the abcdef matrices

use ndarray::{s, Array1, Array2, Array3};
use std::f64::consts::PI;
use crate::mathhelps::trapz;
use crate::dij::dij;
use libm::{j0, j1};

// Holds the output parameter matrices for the main solver
pub struct AbcdefResult {
    pub a: Array2<f64>,
    pub b: Array2<f64>,
    pub cc: Array2<f64>,
    pub cs: Array2<f64>,
    pub drad: Array1<f64>,
    pub dang: Array2<f64>,
    pub e: Array3<f64>,
    pub fcc: Array3<f64>,
    pub fcs: Array3<f64>,
    pub fsc: Array3<f64>,
    pub fss: Array3<f64>,
}

// Rust bisection solver to find the roots of J1(x)
fn find_j1_root(guess: f64) -> f64 {
    let mut low = guess - 1.0;
    let mut high = guess + 1.0;
    
    // 1. Expand the bracket if no sign change is found nearby
    while j1(low) * j1(high) > 0.0 {
        low -= 0.5;
        high += 0.5;
    }
    
    // 2. Perform the Bisection method to pinpoint the root
    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        if (high - low) < 1e-8 {
            return mid;
        }
        if j1(low) * j1(mid) <= 0.0 {
            high = mid;
        } else {
            low = mid;
        }
    }
    (low + high) / 2.0
}

pub fn abcdef(
    lvec: &Array1<i64>,
    r_radius: f64,
    rm_radius: f64,
    r: &Array1<f64>,
    nm: usize,
    ni: usize,
    nq: usize,
    fcr: &Array1<f64>,
    fcphi: &Array1<f64>,
    ur: &Array2<f64>,
    dr: f64,
    dphi: f64,
) -> AbcdefResult {
    let cr = r.len();

    // 1. Generate gam vector (Roots of Bessel J1)
    let mut gam = Array1::<f64>::zeros(ni);
    gam[0] = 0.0;
    let mut x0 = 4.0;

    for i in 1..ni {
        gam[i] = find_j1_root(x0);
        x0 = gam[i] + 3.1;
    }

    // 2. Generate bess matrix: besselj(0, gam(1:ni)*r/R)
    let mut bess = Array2::<f64>::zeros((ni, cr));
    for j in 0..ni {
        for col in 0..cr {
            bess[[j, col]] = j0(gam[j] * r[col] / r_radius);
        }
    }

    // 3. Rebuild coordinates
    let phi_steps = ((2.0 * PI / dphi).round() as usize) + 1;
    let phi = Array1::from_shape_fn(phi_steps, |idx| -PI + (idx as f64) * dphi);

    let crm = ((rm_radius / dr).round() as usize) + 1;
    let rm = Array1::from_shape_fn(crm, |idx| (idx as f64) * dr);

    // 4. Find modal cutoff index (s)
    let mut s_idx = 0;
    if nm > 1 {
        let target_row = if nm < 6 { 1 } else { 5 }; // 0-indexed corresponding to MATLAB's 2 and 6
        let max_val = ur.row(target_row).iter().cloned().fold(f64::MIN, f64::max);
        let threshold = max_val / std::f64::consts::E;
        for idx in 0..cr {
            if ur[[target_row, idx]] >= threshold {
                s_idx = idx;
                break;
            }
        }
    } else {
        s_idx = 1; 
    }

    // 5. Preallocate output arrays
    let mut a = Array2::<f64>::zeros((1, ni));
    let mut b = Array2::<f64>::zeros((1, ni));
    let mut cc = Array2::<f64>::zeros((ni, nq));
    let mut cs = Array2::<f64>::zeros((ni, nq));
    let drad = gam.mapv(|g| (g / r_radius).powi(2));
    let mut dang = Array2::<f64>::zeros((ni, ni));
    let mut e = Array3::<f64>::zeros((ni, ni, nm));

    // 6. Main integration loops for spatial overlap
    for j in 0..ni {
        let bess_j = bess.row(j).to_owned();
        let bess_j_crm = bess_j.slice(s![0..crm]).to_owned();

        a[[0, j]] = (2.0 / rm_radius.powi(2)) * trapz(&rm, &(&bess_j_crm * &rm));
        b[[0, j]] = (2.0 / r_radius.powi(2)) * trapz(r, &(&bess_j * r));

        let bj0 = j0(gam[j]);
        let norm_factor = 2.0 / (r_radius * bj0).powi(2);

        let trapz_r_cc_cs = trapz(r, &(fcr * &bess_j * r));

        for q in 0..nq {
            let q_f64 = q as f64;
            let phi_cos = phi.mapv(|p| (q_f64 * p).cos());
            let phi_sin = phi.mapv(|p| (q_f64 * p).sin());

            let trapz_phi_cos = trapz(&phi, &(fcphi * &phi_cos));
            let trapz_phi_sin = trapz(&phi, &(fcphi * &phi_sin));

            let dij_val = dij(1, (q + 1) as i64);
            let cc_cs_norm = norm_factor / PI / (1.0 + dij_val);

            cc[[j, q]] = cc_cs_norm * trapz_phi_cos * trapz_r_cc_cs;
            cs[[j, q]] = cc_cs_norm * trapz_phi_sin * trapz_r_cc_cs;
        }

        for i in 0..ni {
            let bess_i = bess.row(i).to_owned();
            let j0_i = j0(gam[i]);
            let dang_norm_factor = 2.0 / (r_radius * j0_i).powi(2);

            let r_s = r.slice(s![s_idx..cr]).to_owned();
            let bess_j_s = bess_j.slice(s![s_idx..cr]).to_owned();
            let bess_i_s = bess_i.slice(s![s_idx..cr]).to_owned();

            let dang_integrand = (&bess_j_s / &r_s) * &bess_i_s;
            dang[[i, j]] = dang_norm_factor * trapz(&r_s, &dang_integrand);

            for m in 0..nm {
                let ur_m = ur.row(m).to_owned();
                let e_integrand = &bess_j * &ur_m * &bess_i * r;
                e[[i, j, m]] = dang_norm_factor * trapz(r, &e_integrand);
            }
        }
    }

    // 7. Phase logic matrices
    let mut fcc = Array3::<f64>::zeros((nq, nq, nm));
    let mut fsc = Array3::<f64>::zeros((nq, nq, nm));
    let mut fcs = Array3::<f64>::zeros((nq, nq, nm));
    let mut fss = Array3::<f64>::zeros((nq, nq, nm));

    for k in 0..nq {
        let k_i64 = (k + 1) as i64; 
        for l in 0..nm {
            for q in 0..nq {
                let q_i64 = (q + 1) as i64; 
                
                let kp2l = k_i64 - 1 + 2 * lvec[l];
                let km2l = k_i64 - 1 - 2 * lvec[l];

                let dij_kq = dij(k_i64, q_i64);
                let dij_kp2l_qm1 = dij(kp2l, q_i64 - 1);
                let dij_abs_km2l_qm1 = dij(km2l.abs(), q_i64 - 1);
                let sign_km2l = km2l.signum() as f64;

                fcc[[k, q, l]] = dij_kq + (dij_kp2l_qm1 + dij_abs_km2l_qm1) / 2.0;
                fsc[[k, q, l]] = dij_kq + (dij_kp2l_qm1 + sign_km2l * dij_abs_km2l_qm1) / 2.0;
                fcs[[k, q, l]] = dij_kq - (dij_kp2l_qm1 + dij_abs_km2l_qm1) / 2.0;
                fss[[k, q, l]] = dij_kq - (dij_kp2l_qm1 + sign_km2l * dij_abs_km2l_qm1) / 2.0;
            }
        }
    }

    // Clear initial row boundary (MATLAB fsc(1,:,:) = 0)
    for l in 0..nm {
        for q in 0..nq {
            fsc[[0, q, l]] = 0.0;
            fss[[0, q, l]] = 0.0;
        }
    }

    AbcdefResult { a, b, cc, cs, drad, dang, e, fcc, fcs, fsc, fss }
}