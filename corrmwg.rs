use ndarray::{Array1, Array2, Array3};
use crate::mathhelps::{trapz2d, xcorr2};
use std::time::Instant;

/// Structure to hold the waveguide correlation results
pub struct Corrmwg {
    pub xoff: Array2<f64>,
    pub yoff: Array2<f64>,
    pub cc: Array3<f64>,
    pub cs: Array3<f64>,
}

pub fn corrmwg(
    rff: f64,
    uxyffc: &Array3<f64>,
    uxyffs: &Array3<f64>,
    rwg: f64,
    lvec: &Array1<i32>,
) -> Corrmwg {
    
    let t_start = Instant::now();
    let cxy = uxyffc.shape()[1];
    let nm = uxyffc.shape()[0];

    // 1. Setup global Cartesian grid
    let xff = Array1::linspace(-rff, rff, cxy);
    let yff = Array1::linspace(rff, -rff, cxy); 
    
    // For integration later, we need an ascending y-vector to keep dx positive
    let yff_asc = Array1::linspace(-rff, rff, cxy);

    // 2. Find bounding box for the waveguide
    let mut a_indices = Vec::new();
    for (i, &x) in xff.iter().enumerate() {
        if x.abs() <= rwg {
            a_indices.push(i);
        }
    }
    
    let a1 = a_indices[0].saturating_sub(1);
    let a2 = (a_indices.last().unwrap() + 1).min(cxy - 1);
    let cwg = a2 - a1 + 1;
    
    let xwg = xff.slice(ndarray::s![a1..=a2]).to_owned();
    let ywg = yff.slice(ndarray::s![a1..=a2]).to_owned();
    
    let uxyc = uxyffc.slice(ndarray::s![.., a1..=a2, a1..=a2]).to_owned();
    let uxys = uxyffs.slice(ndarray::s![.., a1..=a2, a1..=a2]).to_owned();

    // 3. Generate Geometric Waveguide Masks 
    let mut wgxyff = Array2::<f64>::zeros((cxy, cxy));
    for i in 0..cxy {
        for j in 0..cxy {
            if (xff[j].powi(2) + yff[i].powi(2)).sqrt() <= rwg {
                wgxyff[[i, j]] = 1.0;
            }
        }
    }

    let mut wgxy = Array2::<f64>::zeros((cwg, cwg));
    for i in 0..cwg {
        for j in 0..cwg {
            if (xwg[j].powi(2) + ywg[i].powi(2)).sqrt() <= rwg {
                wgxy[[i, j]] = 1.0;
            }
        }
    }

    // 4. Setup Offsets Grid
    let max_xwg = xwg.iter().cloned().fold(0./0., f64::max);
    let off_steps = xwg.len() * 2 - 1; // Corresponds to the length resulting from xcorr2
    
    let xoff_vec = Array1::linspace(-2.0 * max_xwg, 2.0 * max_xwg, off_steps);
    let yoff_vec = Array1::linspace(2.0 * max_xwg, -2.0 * max_xwg, off_steps);
    
    let mut xoff = Array2::<f64>::zeros((off_steps, off_steps));
    let mut yoff = Array2::<f64>::zeros((off_steps, off_steps));
    for i in 0..off_steps {
        for j in 0..off_steps {
            xoff[[i, j]] = xoff_vec[j];
            yoff[[i, j]] = yoff_vec[i];
        }
    }

    // 5. Cross-Correlation loop for each mode
    let mut cc = Array3::<f64>::zeros((nm, off_steps, off_steps));
    let mut cs = Array3::<f64>::zeros((nm, off_steps, off_steps));
    
    let center_idx = cwg - 1; // Center peak in the xcorr2 output matrix

    for k in 0..nm {
        let uxyc_k = uxyc.index_axis(ndarray::Axis(0), k).to_owned();
        let cc_2d = xcorr2(&uxyc_k, &wgxy);
        
        let uxyffc_k = uxyffc.index_axis(ndarray::Axis(0), k).to_owned();
        let overlap = &uxyffc_k * &wgxyff;
        
        // Normalization calculation (rp)
        let rp = trapz2d(&xff, &yff_asc, &overlap) / (4.0 * rff.powi(2));
        
        if lvec[k] != 0 {
            let uxys_k = uxys.index_axis(ndarray::Axis(0), k).to_owned();
            let mut cs_2d = xcorr2(&uxys_k, &wgxy);
            let cs_center = cs_2d[[center_idx, center_idx]];
            if cs_center != 0.0 {
                cs_2d.mapv_inplace(|v| v / cs_center * rp);
            }
            cs.index_axis_mut(ndarray::Axis(0), k).assign(&cs_2d);
        }
        
        let mut cc_2d_norm = cc_2d.clone();
        let cc_center = cc_2d_norm[[center_idx, center_idx]];
        if cc_center != 0.0 {
            cc_2d_norm.mapv_inplace(|v| v / cc_center * rp);
        }
        cc.index_axis_mut(ndarray::Axis(0), k).assign(&cc_2d_norm);
    }

    println!("Cross-correlation calculation = {:.2?}s", t_start.elapsed().as_secs_f64());

    Corrmwg {
        xoff,
        yoff,
        cc,
        cs,
    }
}