/// This contains all the parameters used for the somulation!
pub struct MmParams {
    // Constants
    pub q: f64,    // elementary charge [C]
    pub h: f64,    // Planck constant [Js]
    pub kb: f64,   // Boltzmann constant [J/K]
    pub c0: f64,   // vacuum speed of light [m/s]

    // General Physical Parameters
    pub rtc: [f64; 9],        // top mirror reflectivity (c-modes)
    pub rts: [f64; 9],        // top mirror reflectivity (s-modes)
    pub rbc: [f64; 9],        // bottom mirror reflectivity (c-modes)
    pub rbs: [f64; 9],        // bottom mirror reflectivity (s-modes)
    pub alpha_ic: [f64; 9],   // internal losses [cm-1] (c-modes)
    pub alpha_is: [f64; 9],   // internal losses [cm-1] (s-modes)
    pub tau_n: f64,           // carrier-lifetime [s]
    pub tau_esc: f64,         // thermionic emission lifetime [s]
    pub tau_cap: f64,         // ambipolar diffusion time [s]
    pub beta: [f64; 9],       // spontaneous recombination coeff
    pub eta_i: f64,           // current injection efficiency
    pub rs: f64,              // current spreading coefficient [cm]
    pub dn_coeff: f64,        // ambipolar diffusion coeff. [cm2/s]
    pub alpha: f64,           // linewidth enhancement factor

    // Gain Parameters
    pub ntr: f64,             // transparency carrier density
    pub gln: f64,             // logarithmic gain coefficient [cm-1]
    pub g0: f64,              // linear gain coefficient [cm2]
    pub ng: f64,              // group refractive index
    pub epsilon: f64,         // gain compression factor [cm3]
    pub gamma: [f64; 9],      // optical confinement factor

    // Geometrical Parameters
    pub l: f64,               // effective cavity length [cm]
    pub dqw: f64,             // single QW thickness [cm]
    pub nw: f64,              // number of quantum wells
    pub db: f64,              // SCH thickness [cm]
    pub r: f64,               // cavity radius [cm]
    pub rox: f64,             // oxide aperture radius [cm]
    pub t_ox: f64,            // oxide thickness [cm]

    // Thermal Parameters
    pub ne: f64,              // diode voltage parameter
    pub rth: f64,             // thermal resistance K/W
    pub cth: f64,             // thermal capacitance [J/K]
    pub lfp0: f64,            // emission wavelength @300K [nm]
    pub dlfp: f64,            // temperature coeff of lFP [nm/K]
    pub lp0: f64,             // gain peak wavelength @300K
    pub dlp: f64,             // temperature coeff of lp [nm/K]
    pub glw: f64,             // gain profile FWHM [nm]
    pub tref: f64,            // reference temperature [K]

    // Parasitics
    pub rq: f64,              // current source resistance [ohm]
    pub cq: f64,              // current source capacitance [F]
    pub rw: f64,              // bond-wire resistance [ohm]
    pub lw: f64,              // bond-wire inductance [H]
    pub cp: f64,              // pad capacitance [F]
    pub rs_bragg: f64,        // Bragg reflectors resistance [ohm]
    pub ra: f64,              // cavity resistance [ohm]
    pub ca: f64,              // cavity capacitance [F]
}

impl MmParams {
    pub fn new() -> Self {
        // Shared array values
        let ones_9 = [1.0; 9];
        
        // Geometric calculations
        let r: f64 = 8e-4;
        let dqw: f64 = 8e-7;
        let nw: f64 = 3.0;
        let c_spec: f64 = 0.35;
        let rho: f64 = 5.36;
        let cth = rho * c_spec * std::f64::consts::PI * r.powi(2) * nw * dqw;

        Self {
            q: 1.602e-19,
            h: 6.626e-34,
            kb: 1.381e-23,
            c0: 2.998e8,

            rtc: ones_9.map(|x| x * 0.997),
            rts: ones_9.map(|x| x * 0.997),
            rbc: ones_9.map(|x| x * 0.9985),
            rbs: ones_9.map(|x| x * 0.9985),
            alpha_ic: [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 39.2, 40.0, 40.0], // 40 * 0.98 for 7th element
            alpha_is: ones_9.map(|x| x * 40.0),
            tau_n: 2.5e-9,
            tau_esc: 400e-12,
            tau_cap: 45e-12,
            beta: ones_9.map(|x| x * 3e-5),
            eta_i: 1.0,
            rs: 1e-4,
            dn_coeff: 15.0,
            alpha: 2.0,

            ntr: 1.85e18,
            gln: 1500.0,
            g0: 4e-16,
            ng: 4.2,
            epsilon: 5e-17,
            gamma: ones_9.map(|x| x * 0.03),

            l: 900e-7,
            dqw,
            nw,
            db: 40e-7,
            r,
            rox: 6e-4,
            t_ox: 0.7e-6,

            ne: 2e-8,
            rth: 3000.0,
            cth,
            lfp0: 858.0,
            dlfp: 0.06,
            lp0: 848.0,
            dlp: 0.27,
            glw: 40.0,
            tref: 250.0,

            rq: 50.0,
            cq: 0.5e-12,
            rw: 0.4,
            lw: 1e-9,
            cp: 0.5e-12,
            rs_bragg: 20.0,
            ra: 30.0,
            ca: 0.5e-12,
        }
    }
}