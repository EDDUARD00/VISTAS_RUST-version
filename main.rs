mod dij; //Looks for the file name of dij!
mod params;
mod it;
mod corrmwg;
mod mathhelps;
mod injprof;
mod abcdef;
mod fsprop;
mod fmodes;
mod LPcalc;

use ndarray::{s, Array1, Array2, Array3, Axis};
use rand::Rng;
use rand::rng;
//use rand::rngs::ThreadRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{self, Write};
use std::time::Instant;

use crate::abcdef::abcdef;
use crate::dij::dij;
use crate::corrmwg::corrmwg;
use crate::fsprop::fsprop;
use crate::injprof::injprof;
use crate::it::it;
use crate::LPcalc::LPcalc;
use crate::fmodes::fmodes;
use crate::params::Params;

fn main() {
    println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
    println!("%                          V I S T A S                            %");
    println!("%       VCSEL Integrated Spatio-Temporal Advanced Simulator       %");
    println!("%                     RUST ACCELERATED PORT                       %");
    println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");

    // 1. Load Parameters
    let p = Params::new();
    let hvl = p.h * p.c0 / 1e-9;
    let vol = PI * p.R.powi(2) * (p.nw * p.dqw);
    let vg = 100.0 * p.c0 / p.ng;

    // 2. User Inputs
    println!("\nChoose the mode to be feedback: 0=ALL, 1=LP01, 2=LP11, 3=LP21, 4=LP02, 5=LP31");
    print!("Enter mode index for feedback (0-9): ");
    io::stdout().flush().unwrap();
    let mut mode_input = String::new();
    io::stdin().read_line(&mut mode_input).unwrap();
    let mode_choice: usize = mode_input.trim().parse().unwrap_or(0);

    print!("Enter pump factor (e.g., 1.0, 1.3, 1.5): ");
    io::stdout().flush().unwrap();
    let mut pump_input = String::new();
    io::stdin().read_line(&mut pump_input).unwrap();
    let pump_val: f64 = pump_input.trim().parse().unwrap_or(1.0);

    let mode_labels = [
        "ALL", "LP01", "LP11", "LP21", "LP02", "LP31", "LP12", "LP41", "LP22", "LP03",
    ];
    let selected_mode_name = mode_labels[mode_choice];

    let ni = 15;
    let dx = 0.0;
    let dy = 0.0;
    let runs = [1]; // Set to [1] for testing, expand to [1,2..10] for production
    let feedback_vec = [0.25]; //[0.0, 0.25, 0.50];
    let noise_factor_vec = [0.0]; //[0.0, 0.001, 0.01];

    let total_tstart = Instant::now();

    for &run in &runs {
        for &fb in &feedback_vec {
            for &nf in &noise_factor_vec {
                let noise_factor = nf;
                println!("\n--- Executing: Mode={}, Pump={}, FB={}, Noise={}, Run={} ---", 
                         selected_mode_name, pump_val, fb, noise_factor, run);

                // Derived Efficiency Parameters
                let mut alpha_mc = Array1::<f64>::zeros(15);
                let mut alpha_ms = Array1::<f64>::zeros(15);
                let mut tau_sc = Array1::<f64>::zeros(15);
                let mut tau_ss = Array1::<f64>::zeros(15);
                let mut eta_optc = Array1::<f64>::zeros(15);
                let mut eta_opts = Array1::<f64>::zeros(15);

                for m in 0..9 {
                    alpha_mc[m] = (1.0 / p.l) * (1.0 / (p.Rtc[m] * p.Rbc[m]).sqrt()).ln();
                    alpha_ms[m] = (1.0 / p.l) * (1.0 / (p.Rts[m] * p.Rbs[m]).sqrt()).ln();
                    tau_sc[m] = 1.0 / (vg * (alpha_mc[m] + p.alpha_ic[m]));
                    tau_ss[m] = 1.0 / (vg * (alpha_ms[m] + p.alpha_is[m]));
                    
                    let fc = (1.0 - p.Rtc[m]) / ((1.0 - p.Rtc[m]) + (p.Rtc[m] / p.Rbc[m]).sqrt() * (1.0 - p.Rbc[m]));
                    let fs = (1.0 - p.Rts[m]) / ((1.0 - p.Rts[m]) + (p.Rtc[m] / p.Rbs[m]).sqrt() * (1.0 - p.Rbs[m]));
                    
                    eta_optc[m] = fc * alpha_mc[m] / (p.alpha_ic[m] + alpha_mc[m]);
                    eta_opts[m] = fs * alpha_ms[m] / (p.alpha_is[m] + alpha_ms[m]);
                }

                // Grid Parameters
                let cr = (p.R / p.dr).round() as usize;
                let r_vec = Array1::from_shape_fn(cr + 1, |i| (i as f64) * p.dr);

                // External Modules
                let current = it(pump_val, p.dt);
                let prof = injprof(&r_vec, p.R, p.Rox, p.rs, p.dphi);
                
                let lfp0_cm = p.lfp0 * 1e-7;
                let modes = LPcalc(lfp0_cm, p.nc, p.dn, p.Rm, p.R, cr);
                let nm = modes.nm;
                let ur_squared = modes.ur.mapv(|x| x.powi(2));
                
                let lvec_full: [i64; 14] = [0, 1, 2, 0, 3, 1, 4, 2, 0, 5, 3, 6, 1, 4];
                let lvec = Array1::from_vec(lvec_full[0..nm].to_vec());
                let nq = 1 + 2 * *lvec.iter().max().unwrap_or(&0) as usize;

                let mut ur_norm = Array2::<f64>::zeros((nm, cr + 1));
                let mut lzer = Array1::<f64>::ones(nm);
                for k in 0..nm {
                    let trap_val = mathhelps::trapz(&r_vec, &(&ur_squared.row(k).to_owned() * &r_vec));
                    let dij_val = dij::dij(0, lvec[k] as i64);
                    let norm = p.R.powi(2) / (1.0 + dij_val) / trap_val;
                    for j in 0..=cr { ur_norm[[k, j]] = ur_squared[[k, j]] * norm; }
                    if lvec[k] == 0 { lzer[k] = 0.0; }
                }

                // Far Field Propagation
                let fsp = fsprop(&modes.ur, lfp0_cm, p.dext, cr, 300, p.R, p.dr, &lvec);
                let corr = corrmwg(fsp.rff, &fsp.uxyffc, &fsp.uxyffs, p.rwg, &lvec);

                // Find misalignment coordinates
                let coordx = corr.xoff.row(0).iter().position(|&x| x > dx).unwrap_or(0);
                let coordy = corr.yoff.column(0).iter().position(|&y| y < dy).unwrap_or(0);

                let mut eta_extc = Array1::<f64>::zeros(nm);
                let mut eta_exts = Array1::<f64>::zeros(nm);
                for m in 0..nm {
                    eta_extc[m] = (1.0 - p.Rext) * corr.cc[[m, coordx, coordy]];
                    eta_exts[m] = (1.0 - p.Rext) * corr.cs[[m, coordx, coordy]];
                }

                // Space-Time Preallocations
                let ct = current.ct;
                let mut nc = Array3::<f64>::zeros((ni, nq, ct + 1));
                let mut ns = Array3::<f64>::zeros((ni, nq, ct + 1));
                let mut na = Array1::<f64>::zeros(ct + 1);
                
                let mut sc = Array2::<f64>::zeros((nm, ct + 1));
                let mut ss = Array2::<f64>::zeros((nm, ct + 1));
                let mut fc = Array2::<f64>::zeros((nm, ct + 1));
                let mut fs = Array2::<f64>::zeros((nm, ct + 1));

                // External Cavity Setup
                let tau_l = 2.0 * p.l / vg;
                let ctext = (2.0 * p.Lext * p.next / 100.0 / p.c0 / p.dt).round() as usize;
                let tetaext = (ctext as f64 * p.dt) * 2.0 * PI * p.c0 / p.lfp0 / 1e-9;
                
                let mut mask = Array1::<f64>::zeros(nm);
                if mode_choice == 0 { mask.fill(1.0); } 
                else if mode_choice - 1 < nm { mask[mode_choice - 1] = 1.0; }

                let mut kcext_fb = Array1::<f64>::zeros(nm);
                let mut ksext_fb = Array1::<f64>::zeros(nm);
                for m in 0..nm {
                    kcext_fb[m] = fb * (eta_extc[m] * (1.0 - p.Rtc[m]) * (p.Rext / p.Rtc[m]).sqrt() / tau_l) * mask[m];
                    ksext_fb[m] = fb * (eta_exts[m] * (1.0 - p.Rts[m]) * (p.Rext / p.Rts[m]).sqrt() / tau_l) * mask[m];
                }

                // Spatial Overlap Integrals
                let mut abc = abcdef(&lvec, p.R, p.Rm, &r_vec, nm, ni, nq, &prof.fcr, &prof.fcphi, &ur_norm, p.dr, p.dphi);
                
                let mut b_mat = Array2::<f64>::zeros((nm, ni));
                for m in 0..nm {
                    for j in 0..ni {
                        b_mat[[m, j]] = (p.Gamma[m] * p.beta[m] / p.tau_n) * abc.b[[0, j]];
                    }
                }
                
                abc.cc.mapv_inplace(|v| v * p.eta_i / p.q / vol);
                abc.cs.mapv_inplace(|v| v * p.eta_i / p.q / vol);
                abc.drad.mapv_inplace(|v| p.DN * v + 1.0 / p.tau_n);
                abc.dang.mapv_inplace(|v| p.DN * v);
                abc.e.mapv_inplace(|v| vg * v / 2.0);

                let mut q2 = Array2::<f64>::zeros((ni, nq));
                for j in 0..ni { for q in 0..nq { q2[[j, q]] = (q as f64).powi(2); } }

                let fn_noise = (2.0 / vol / p.tau_n / p.dt).sqrt();
                let mut nthc = Array1::<f64>::zeros(nm);
                let mut nths = Array1::<f64>::zeros(nm);
                for m in 0..nm {
                    nthc[m] = p.Ntr * (1.0 / (vg * tau_sc[m] * p.Gamma[m] * p.gln)).exp();
                    nths[m] = p.Ntr * (1.0 / (vg * tau_ss[m] * p.Gamma[m] * p.gln)).exp();
                }
                let a_factor = p.alpha / 2.0;

                // RNG Setup
                let mut rng= rng();
                let normal = Normal::new(0.0, 1.0).unwrap();

                // --- MAIN SPATIO-TEMPORAL LOOP ---
                println!("Starting time loop...");
                let loop_start = Instant::now();

                for i in 0..ct {
                    let nctmp = nc.slice(s![.., .., i]).to_owned();
                    let nstmp = ns.slice(s![.., .., i]).to_owned();
                    let sctmp = sc.column(i).to_owned();
                    let sstmp = ss.column(i).to_owned();

                    let rspc = b_mat.dot(&nctmp.column(0).to_owned());
                    let rsps = &rspc * &lzer;

                    // Langevin Noise
                    let xn = normal.sample(&mut rng);
                    let xsc = Array1::from_shape_fn(nm, |_| normal.sample(&mut rng));
                    let xss = Array1::from_shape_fn(nm, |_| normal.sample(&mut rng));
                    let xfc = Array1::from_shape_fn(nm, |_| normal.sample(&mut rng));
                    let xfs = Array1::from_shape_fn(nm, |_| normal.sample(&mut rng));

                    let mut fsc = Array1::<f64>::zeros(nm);
                    let mut fss = Array1::<f64>::zeros(nm);
                    let mut ffc = Array1::<f64>::zeros(nm);
                    let mut ffs = Array1::<f64>::zeros(nm);

                    for m in 0..nm {
                        fsc[m] = noise_factor * (2.0 * rspc[m] * sctmp[m] / p.dt).sqrt() * xsc[m];
                        fss[m] = noise_factor * (2.0 * rsps[m] * sstmp[m] / p.dt).sqrt() * xss[m];
                        
                        let safe_xsc = if xsc[m].abs() > 1e-10 { xsc[m] } else { 1e-10 };
                        let safe_xss = if xss[m].abs() > 1e-10 { xss[m] } else { 1e-10 };
                        
                        ffc[m] = noise_factor * 0.5 * fsc[m] / (sctmp[m] + 1.0) / safe_xsc * xfc[m];
                        ffs[m] = noise_factor * 0.5 * fss[m] / (sstmp[m] + 1.0) / safe_xss * xfs[m];
                    }
                    let fn_tot = noise_factor * (-fsc.sum() - fss.sum() + fn_noise * nctmp[[0, 0]].max(0.0).sqrt() * xn);

                    // Gain Matrices
                    let g0 = if na[i] > p.Ntr { p.gln * ((na[i] + 1.0) / p.Ntr).ln() / (na[i] - p.Ntr) } else { p.g0 };
                    
                    let mut gcc = Array3::<f64>::zeros((nm, ni, nq));
                    let mut gcs = Array3::<f64>::zeros((nm, ni, nq));
                    let mut gsc = Array3::<f64>::zeros((nm, ni, nq));
                    let mut gss = Array3::<f64>::zeros((nm, ni, nq));
                    let mut gccf = Array1::<f64>::zeros(nm);
                    let mut gcsf = Array1::<f64>::zeros(nm);

                    for m in 0..nm {
                        let etmp = &abc.e.slice(s![.., .., m]).to_owned() * g0;
                        let fcctmp = abc.fcc.slice(s![.., .., m]).to_owned();
                        let fcstmp = abc.fcs.slice(s![.., .., m]).to_owned();
                        
                        let gcctmp = etmp.dot(&nctmp).dot(&fcctmp);
                        let gcstmp = etmp.dot(&nctmp).dot(&fcstmp);

                        gccf[m] = gcctmp[[0,0]] - etmp[[0,0]] * fcctmp[[0,0]] * nthc[m];
                        gcsf[m] = gcstmp[[0,0]] - etmp[[0,0]] * fcstmp[[0,0]] * nths[m];

                        let etmp_c0 = etmp.column(0).to_owned();
                        
                        for j in 0..ni {
                            for q in 0..nq {
                                gcc[[m, j, q]] = gcctmp[[j, q]] - etmp_c0[j] * fcctmp[[0, q]] * p.Ntr;
                                gcs[[m, j, q]] = gcstmp[[j, q]] - etmp_c0[j] * fcstmp[[0, q]] * p.Ntr;
                            }
                        }
                        
                        let gsctmp = etmp.dot(&nstmp).dot(&abc.fsc.slice(s![.., .., m]).to_owned());
                        let gsstmp = etmp.dot(&nstmp).dot(&abc.fss.slice(s![.., .., m]).to_owned());
                        gsc.slice_mut(s![m, .., ..]).assign(&gsctmp);
                        gss.slice_mut(s![m, .., ..]).assign(&gsstmp);
                    }

                    // Gain Compression
                    for m in 0..nm {
                        let compr_c = 1.0 + p.epsilon * sctmp[m];
                        let compr_s = 1.0 + p.epsilon * sstmp[m];
                        
                        gccf[m] /= compr_c;
                        gcsf[m] /= compr_s;
                        
                        let s2c = sctmp[m] / compr_c;
                        let s2s = sstmp[m] / compr_s;
                        
                        gcc.slice_mut(s![m, .., ..]).mapv_inplace(|v| v * s2c);
                        gcs.slice_mut(s![m, .., ..]).mapv_inplace(|v| v * s2s);
                        gsc.slice_mut(s![m, .., ..]).mapv_inplace(|v| v * s2c);
                        gss.slice_mut(s![m, .., ..]).mapv_inplace(|v| v * s2s);
                    }

                    // Spatial Integration Sums
                    let mut sum_gcc_gcs = Array2::<f64>::zeros((ni, nq));
                    let mut sum_gsc_gss = Array2::<f64>::zeros((ni, nq));
                    for m in 0..nm {
                        sum_gcc_gcs = sum_gcc_gcs + &gcc.slice(s![m, .., ..]) + &gcs.slice(s![m, .., ..]);
                        sum_gsc_gss = sum_gsc_gss + &gsc.slice(s![m, .., ..]) + &gss.slice(s![m, .., ..]);
                    }

                    // Euler Differential Update Step
                    let dang_nctmp = abc.dang.dot(&nctmp);
                    let dang_nstmp = abc.dang.dot(&nstmp);

                    for j in 0..ni {
                        let dr_val = abc.drad[j];
                        for q in 0..nq {
                            nc[[j, q, i + 1]] = nctmp[[j, q]] + p.dt * (abc.cc[[j, q]] * current.i0[i] - nctmp[[j, q]] * dr_val - dang_nctmp[[j, q]] * q2[[j, q]] - sum_gcc_gcs[[j, q]]);
                            ns[[j, q, i + 1]] = nstmp[[j, q]] + p.dt * (abc.cs[[j, q]] * current.i0[i] - nstmp[[j, q]] * dr_val - dang_nstmp[[j, q]] * q2[[j, q]] - sum_gsc_gss[[j, q]]);
                        }
                    }

                    nc[[0, 0, i + 1]] += p.dt * fn_tot;
                    if nc[[0, 0, i + 1]] < 0.0 { nc[[0, 0, i + 1]] = 0.0; }
                    na[i + 1] = abc.a[[0, 0]] * nc[[0, 0, i + 1]];

                    for m in 0..nm {
                        sc[[m, i + 1]] = sctmp[m] + p.dt * (-sctmp[m] / tau_sc[m] + rspc[m] + p.Gamma[m] * gcc[[m, 0, 0]] + fsc[m]);
                        ss[[m, i + 1]] = sstmp[m] + p.dt * (-sstmp[m] / tau_ss[m] + rsps[m] + p.Gamma[m] * gcs[[m, 0, 0]] + fss[m]);
                        
                        fc[[m, i + 1]] = fc[[m, i]] + p.dt * (a_factor * p.Gamma[m] * gccf[m] + ffc[m]);
                        fs[[m, i + 1]] = fs[[m, i]] + p.dt * (a_factor * p.Gamma[m] * gcsf[m] + ffs[m]);
                    }

                    // Optical Feedback Delay Equations
                    if i > ctext {
                        for m in 0..nm {
                            let idx_delay = i - ctext;
                            let tetac = fc[[m, i]] - fc[[m, idx_delay]] + tetaext;
                            let tetas = fs[[m, i]] - fs[[m, idx_delay]] + tetaext;

                            sc[[m, i + 1]] += p.dt * (kcext_fb[m] * (sc[[m, idx_delay]] * sctmp[m]).max(0.0).sqrt() * tetac.cos() * 2.0);
                            ss[[m, i + 1]] += p.dt * (ksext_fb[m] * (ss[[m, idx_delay]] * sstmp[m]).max(0.0).sqrt() * tetas.cos() * 2.0);

                            fc[[m, i + 1]] -= p.dt * (kcext_fb[m] * (sc[[m, idx_delay]] / (sctmp[m] + 1.0)).max(0.0).sqrt() * tetac.sin());
                            fs[[m, i + 1]] -= p.dt * (ksext_fb[m] * (ss[[m, idx_delay]] / (sstmp[m] + 1.0)).max(0.0).sqrt() * tetas.sin());
                        }
                    }

                    // Boundary enforcement (Photons cannot be negative)
                    for m in 0..nm {
                        sc[[m, i + 1]] = (sc[[m, i + 1]] + sc[[m, i + 1]].abs()) / 2.0;
                        ss[[m, i + 1]] = (ss[[m, i + 1]] + ss[[m, i + 1]].abs()) / 2.0;
                    }
                }
                println!("Main loop calculation = {:.2?}s", loop_start.elapsed().as_secs_f64());

                // --- SAVE DATA TO CSV ---
                let filename = format!("_selectivefeedback_{}_pump={:.1}_noise={:.3}_feedback={:.2}_run={}.csv", 
                                        selected_mode_name, pump_val, noise_factor, fb, run);
                let mut file = File::create(&filename).unwrap();
                
                // Write Headers
                write!(file, "Time,").unwrap();
                for m in 0..nm { write!(file, "{},", mode_labels[m + 1]).unwrap(); }
                writeln!(file, "Ptot,Na,Ss").unwrap();

                // Compute power scaling parameters
                let mut kc = Array1::<f64>::zeros(nm);
                let mut ks = Array1::<f64>::zeros(nm);
                for m in 0..nm {
                    kc[m] = vol * hvl / p.lfp0 * eta_optc[m] / tau_sc[m] / p.Gamma[m];
                    ks[m] = vol * hvl / p.lfp0 * eta_opts[m] / tau_ss[m] / p.Gamma[m];
                }

                // Write rows
                for i in 0..=ct {
                    let time_ns = current.t[i] * 1e9;
                    write!(file, "{:.6},", time_ns).unwrap();
                    
                    let mut ptot = 0.0;
                    let mut stot = 0.0;
                    for m in 0..nm {
                        let pc = eta_extc[m] * kc[m] * sc[[m, i]] * 1e3;
                        let ps = eta_exts[m] * ks[m] * ss[[m, i]] * 1e3;
                        let p_mode = pc + ps;
                        ptot += p_mode;
                        stot += sc[[m, i]] + ss[[m, i]];
                        write!(file, "{:.6},", p_mode).unwrap();
                    }
                    writeln!(file, "{:.6},{:.6},{:.6}", ptot, na[i], stot).unwrap();
                }
                println!("Saved results to: {}", filename);
            }
        }
    }
    println!("Total execution time: {:.2?}s", total_tstart.elapsed().as_secs_f64());
}