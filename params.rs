/// This contains all the parameters used for the somulation!

pub struct Params {
    // Constants
    pub q: f64,    // elementary charge [C]
    pub h: f64,    // Planck constant [Js]
    pub kb: f64,   // Boltzmann constant [J/K]
    pub c0: f64,   // vacuum speed of light [m/s]

    // General Physical Parameters
    pub Rtc: [f64; 15],        // top mirror reflectivity (c-modes)
    pub Rts: [f64; 15],        // top mirror reflectivity (s-modes)
    pub Rbc: [f64; 15],        // bottom mirror reflectivity (c-modes)
    pub Rbs: [f64; 15],        // bottom mirror reflectivity (s-modes)
    pub alpha_ic: [f64; 15],   // internal losses [cm-1] (c-modes)
    pub alpha_is: [f64; 15],   // internal losses [cm-1] (s-modes)
    pub tau_n: f64,           // carrier-lifetime [s]
    pub tau_esc: f64,         // thermionic emission lifetime [s]
    pub tau_cap: f64,         // ambipolar diffusion time [s]
    pub beta: [f64; 15],       // spontaneous recombination coeff
    pub eta_i: f64,           // current injection efficiency
    pub rs: f64,              // current spreading coefficient [cm]
    pub DN: f64,        // ambipolar diffusion coeff. [cm2/s]
    pub alpha: f64,           // linewidth enhancement factor

    // Gain Parameters
    pub Ntr: f64,             // transparency carrier density
    pub gln: f64,             // logarithmic gain coefficient [cm-1]
    pub g0: f64,              // linear gain coefficient [cm2]
    pub ng: f64,              // group refractive index
    pub epsilon: f64,         // gain compression factor [cm3]
    pub Gamma: [f64; 15],      // optical confinement factor

    // Geometrical Parameters
    pub l: f64,               // effective cavity length [cm]
    pub dqw: f64,             // single QW thickness [cm]
    pub nw: f64,              // number of quantum wells
    pub db: f64,              // SCH thickness [cm]
    pub R: f64,               // cavity radius [cm]
    pub Rox: f64,             // oxide aperture radius [cm]
    pub t: f64,               // oxide thickness [cm]
    
    //modal parameters
    pub nc: f64,              // core equivalent refractive index
    pub dn: f64,              // equivalent fractional refractive index change
    pub Rm: f64,              // equivalent core radius [cm]

    // Thermal Parameters
    pub Ne: f64,              // diode voltage parameter
    pub Rth: f64,             // thermal resistance K/W
    pub Cth: f64,             // thermal capacitance [J/K]
    pub lfp0: f64,            // emission wavelength @300K [nm]
    pub dlfp: f64,            // temperature coeff of lFP [nm/K]
    pub lp0: f64,             // gain peak wavelength @300K
    pub dlp: f64,             // temperature coeff of lp [nm/K]
    pub glw: f64,             // gain profile FWHM [nm]
    pub Tref: f64,            // reference temperature [K]
    pub Il0: f64,             // leakage parameter [A]
    pub a0: f64,              // leakage parameter
    pub a1: f64,              // leakage Parameters
    pub a2: f64,              // leakage parameter
    pub a3: f64,              // leakage parameter
    pub b1: f64,              // leakage parameter

    // Parasitics
    pub Rq: f64,              // current source resistance [ohm]
    pub Cq: f64,              // current source capacitance [F]
    pub Rw: f64,              // bond-wire resistance [ohm]
    pub Lw: f64,              // bond-wire inductance [H]
    pub Cp: f64,              // pad capacitance [F]
    pub Rs: f64,              // Bragg reflectors resistance [ohm]
    pub Ra: f64,              // cavity resistance [ohm]
    pub Ca: f64,              // cavity capacitance [F]
    
    //Waveguide Parameters
    pub dext: f64,            // distance of laser facet -fiber cm
    pub Lext: f64,            // External cavity length cm
    pub next: f64,            // external medium refractive index
    pub Rext: f64,            // external power reflectance1
    pub rwg: f64,             // fiber core radius
    
    //finite differences Parameters
    pub dt: f64,              // time steps
    pub dr: f64,              // radial steps
    pub dphi: f64,            // aximuthal steps
    
}

impl Params {
    pub fn new() -> Self {
        // Shared array values
        let ones_15 = [1.0; 15];
        
        // Geometric calculations
        let R: f64 = 8e-4;
        let dqw: f64 = 8e-7;
        let nw: f64 = 3.0;
        let c: f64 = 0.35;
        let rho: f64 = 5.36;
        let Rox: f64 = 6e-4;
        let nc: f64 = 3.6; 
        let next: f64 = 1.46;
        
        
        let Rext = ((1.0 - next) / (1.0 + next)).powi(2);
        let dn = 0.025/nc;
        let Cth = rho * c * std::f64::consts::PI * R.powi(2) * nw * dqw;
        let Rm = Rox;
        let dr = R / (40.0 * R / Rm).floor();
        let dphi =  2.0 * std::f64::consts::PI / 35.0;
        

        Self {
            q: 1.602e-19,
            h: 6.626e-34,
            kb: 1.381e-23,
            c0: 2.998e8,

            Rtc: ones_15.map(|x| x * 0.997),
            Rts: ones_15.map(|x| x * 0.997),
            Rbc: ones_15.map(|x| x * 0.9985),
            Rbs: ones_15.map(|x| x * 0.9985),
            alpha_ic: [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 39.2, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0], // 40 * 0.98 for 7th element
            alpha_is: ones_15.map(|x| x * 40.0),
            tau_n: 2.5e-9,
            tau_esc: 400e-12,
            tau_cap: 45e-12,
            beta: ones_15.map(|x| x * 3e-5),
            eta_i: 1.0,
            rs: 1e-4,
            DN: 15.0,
            alpha: 2.0,

            Ntr: 1.85e18,
            gln: 1500.0,
            g0: 4e-16,
            ng: 4.2,
            epsilon: 5e-17,
            Gamma: ones_15.map(|x| x * 0.03),

            l: 900e-7,
            dqw: 8e-7,
            nw,
            db: 40e-7,
            R:8e-4,
            Rox: 6e-4,
            t: 0.7e-6,
            
            nc, 
            dn,
            Rm, 

            Ne: 2e-8,
            Rth: 3000.0,
            Cth,
            lfp0: 858.0,
            dlfp: 0.06,
            lp0: 848.0,
            dlp: 0.27,
            glw: 40.0,
            Tref: 250.0,
            Il0: 6e-4,
            a0: -700.0,
            a1: 5.4e-17,
            a2: 2.4e-19,
            a3: -3.4e21,
            b1: 0.5e16,
            

            Rq: 50.0,
            Cq: 0.5e-12,
            Rw: 0.4,
            Lw: 1e-9,
            Cp: 0.5e-12,
            Rs: 20.0,
            Ra: 30.0,
            Ca: 0.5e-12,
            
            dext: 30e-4,
            Lext: 30.0,
            next,
            Rext,
            rwg: 25e-4,
            
            dt: 1e-12,
            dr,
            dphi,
        }
    }
}