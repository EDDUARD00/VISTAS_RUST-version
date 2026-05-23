//This conatins the current injected into the VCSEL 

use ndarray::Array1;

pub struct IT {
  pub ct: usize,
  pub ctinit: usize,
  pub t: Array1<f64>,
  pub tb: f64,
  pub i0: Array1<f64>,
  pub ion: f64, 
  pub pump: f64,
}

pub fn it(pump:f64, dt: f64) -> IT {
  let t_simulated = 1e-9 * 100.0;
  let ct = (t_simulated / dt).round() as usize;
  let ctinit: usize = 0;
  let tb = 0.0;
  let ith = 2.138;
  let pump = pump;
  
  let t = Array1::from_shape_fn(ct + 1, |idx| (idx as f64) * dt);
  let ion = ith * 1e-3 * pump;
  let mut i0 = Array1::from_elem(ct + 1, ion);
  i0[0] = 0.0;
  
  //This are the outputs of hte it.rs hehe
  IT{
    ct, 
    ctinit,
    t,
    tb,
    ion,
    i0,
    pump,
  }
}