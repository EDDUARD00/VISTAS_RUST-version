mod dij; //Looks for the file name of dij!
mod params;
mod it;
mod corrmwg;
mod mathhelps;
mod injprof;
mod abcdef;
mod fsprop;
mod fmodes;


use crate::dij::dij; // uses the function inside the dij file
use crate::params::Params;
use crate::it::{it, IT};
use ndarray::{Array1, Array3, Array2};
use std::f64::consts::PI;
use crate::injprof::injprof;
use crate::corrmwg::corrmwg;
use crate::abcdef::abcdef;
use crate::fsprop::fsprop;
use crate::fmodes::fmodes;






//sample run
fn main(){
  //let  delta_match = dij(3,3);
  //let mismatch = dij(3,5);
  let pump = 1.2; 
  let p = Params::new(); //initiate parameter structures 
  let current = it(pump, p.dt);
  
  println!("Total simulation time is : {}", current.ct);
  println!("Simulation pump {}", current.ion);
  println!("------------------------------------------------------------------------------");
  // ------------------------------------------------------------------------------
  // Checking if corrmwg is working!!!
  // 1. Setup minimal mock data for the test
    let nm = 2;       // Test with 2 modes
    let cxy = 11;     // Transverse elements (keep it small for a fast test)
    let rff = 50e-4;  // Far-field radius [cm]
    let rwg = 25e-4;  // Waveguide radius [cm]
    
    // Mock far-field intensity profiles (filling with 1.0s to simulate data)
    let uxyffc = Array3::<f64>::ones((nm, cxy, cxy));
    let uxyffs = Array3::<f64>::ones((nm, cxy, cxy));
    
    // Mock azimuthal order vector (LP01 and LP11)
    let lvec = Array1::from_vec(vec![0, 1]); 

    // 2. Execute the cross-correlation function
    let result = corrmwg(rff, &uxyffc, &uxyffs, rwg, &lvec);

    // 3. Print the results to verify functionality
    let off_rows = result.xoff.nrows();
    let off_cols = result.xoff.ncols();
    
    println!("Success! Cross-correlation matrix generated without panicking.");
    println!("Offset grid size: {} x {}", off_rows, off_cols);
    println!("Cc matrix shape: {:?}", result.cc.shape());
    println!("Cs matrix shape: {:?}", result.cs.shape());
    
    // Print the center peak value of the correlation matrix for the first mode
    let center_idx = off_rows / 2;
    println!("Center value of Cc for Mode 0: {:.6}", result.cc[[0, center_idx, center_idx]]);
    // ------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------
    println!("------------------------------------------------------------------------------");
    println!("--- VISTAS Debug: Testing injprof ---");

    // 1. Setup mock parameters from your MM_PARAMS
    let r_cavity:f64 = 8e-4; // R [cm]
    let rox:f64 = 6e-4;      // Rox [cm]
    let rs:f64 = 1e-4;       // rs [cm]
    let dphi:f64 = 2.0 * PI / 35.0; // 35 azimuthal steps
    
    // 2. Mock radial vector (r)
    let rm = 6e-4; // Equivalent core radius
    let dr = p.R / (40.0 * p.R / rm).floor();
    let cr = (p.R / dr).round() as usize;
    let r_vec = Array1::from_shape_fn(cr + 1, |i| (i as f64) * dr);

    // 3. Execute the function
    let profile = injprof(&r_vec, p.R, p.Rox, rs, p.dphi);

    println!("Success! Current distribution computed.");
    println!("fcr shape: {:?}", profile.fcr.len());
    println!("fcphi shape: {:?}", profile.fcphi.len());
    println!("Center fcr value: {:.6}", profile.fcr[0]);
    println!("Edge fcr value (decaying): {:.6}", profile.fcr[profile.fcr.len() - 1]);
    
    // ------------------------------------------------------------------------------
    println!("------------------------------------------------------------------------------");
    println!("--- VISTAS Debug: Testing abcdef with params.rs ---");
    
    // Set the simulation resolutions matching your automated VCSEL setup
    let ni = 15; // Radial resolution 
    let nq = 5;  // Azimuthal resolution
    let nm = 2;  // Testing with 2 modes (e.g., LP01 and LP11)

    // 2. Build the radial vector (r) dynamically
    let cr = (p.R / p.dr).round() as usize; 
    let r_vec = Array1::from_shape_fn(cr + 1, |i| (i as f64) * p.dr);

    // 3. Mock the current spreading functions and modal intensities
    // We fill these with 1.0s just to verify the mathematical engine can process the shapes
    let fcr = Array1::<f64>::ones(cr + 1);
    
    let phi_steps = ((2.0 * std::f64::consts::PI / p.dphi).round() as usize) + 1;
    let fcphi = Array1::<f64>::ones(phi_steps);
    
    let ur = Array2::<f64>::ones((nm, cr + 1));
    let lvec = Array1::from_vec(vec![0, 1]); // Azimuthal order vector

    // 4. Execute the abcdef spatial overlap and phase integration engine!
    let results = abcdef(&lvec, p.R, p.Rm, &r_vec, nm, ni, nq, &fcr, &fcphi, &ur, p.dr, p.dphi);

    // 5. Print out the matrix shapes to verify compilation and execution
    println!("Success! Spatial overlap parameters computed without panicking.");
    println!("Drad vector length: {}", results.drad.len());
    println!("Dang matrix shape: {:?}", results.dang.shape());
    println!("E overlap matrix shape: {:?}", results.e.shape());
    println!("Phase fsc matrix shape: {:?}", results.fsc.shape());
    
    // Print a sample mathematical value to ensure the integration ran correctly
    println!("Sample value of a[0, 0]: {:.6}", results.a[[0, 0]]);
 // println!("Oxide aperture size is {}", p.Rext);
//  println!("dij(3,3) = {}", delta_match);
//  println!("dij(3,5) = {}", mismatch);
    println!("------------------------------------------------------------------------------");
    println!("--- VISTAS Debug: Testing fsprop ---");
    
    // Use smaller values for the test to keep compilation output fast
    let cxy = 111; // Cartesian transverse elements
    let nm = 2;    // Number of modes testing
    let cr = (p.R / p.dr).round() as usize;

    let lambda = 858.0 * 1e-7; // Wavelength [cm]
    let d = 30e-4;             // Propagation distance [cm]
    
    // Mock the modal intensity data (ur)
    let mut ur = Array2::<f64>::zeros((nm, cr + 1));
    for i in 0..=cr {
        // Create a fake decaying gaussian mode to feed the FFT
        ur[[0, i]] = (-(i as f64) / (cr as f64)).exp(); 
        ur[[1, i]] = (-(i as f64) / (cr as f64)).exp(); 
    }
    
    let lvec = Array1::from_vec(vec![0, 1]); // Azimuthal modes

    // Execute Far-Field FFT Engine
    let result = fsprop(&ur, lambda, d, cr, cxy, p.R, p.dr, &lvec);

    println!("Success! FFT Spatial propagation calculated.");
    println!("Uxyffc shape: {:?}", result.uxyffc.shape());
    println!("Calculated Far-Field Radius (Rff): {:.6} cm", result.rff);
    
    let center = cxy / 2;
    println!("Center Far-Field Intensity (Mode 0): {:.6}", result.uxyffc[[0, center, center]]);
}