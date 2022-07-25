pub struct WeightFunction {
    pub function_name: String,
    pub parameters: Vec<f64>
}

impl WeightFunction {

    pub fn new(function_name: String, parameters: Vec<f64>) -> Self {

        match function_name.as_str() {

            "dagum" => {
                assert!(parameters.len() == 3, "For function \"{}\" there must be exactly 3 parameters!", function_name);
                for &item in &parameters {
                    assert!(item > 0., "For function \"{}\" all parameters mut be positive!", function_name);
                }
            }

            "hyper_exp" => {
                assert!(parameters.len() % 2 == 0, "For function \"{}\" there must be an even number of parameters!", function_name);
                for &item in &parameters {
                    assert!(item > 0., "For function \"{}\" all parameters mut be positive!", function_name);
                }
            },

            other => unimplemented!("No function implemented with name \"{}\"!", other)
        }

        Self { function_name, parameters }
    }

    pub fn integral(&self, x: Option<f64>) -> f64 {
        match self.function_name.as_str() {
            "hyper_exp" => self.hyper_exp(x),
            "dagum" => self.dagum(x),
            other => panic!("Unable to identify function {}!", other)
        }
    }

    /// Calculates the hyper-exponential distribution CDF at a given point x.
    /// 
    /// The parameters are in the form of (A_1, A_2, ... A_n, B_1, B_2, ... B_n) and the
    /// CDF is calculated as 1 - SUM_i(A_i * exp(-B_i * x)) / SUM_i(A_i).
    fn hyper_exp(&self, x: Option<f64>) -> f64 {
        
        if x == None {
            return 1.0;
        }
        let x = x.unwrap();

        let mut norm = 0.;
        let mut sum = 0.;

        let (params_a, params_b) = {
            let border_idx = self.parameters.len() / 2;
            (&self.parameters[..border_idx], &self.parameters[border_idx..])
        };
    
        for (&item_a, &item_b) in params_a.iter().zip(params_b.iter()) {
            sum += item_a * (- item_b * x).exp();
            norm += item_a;
        }
    
        1. - sum / norm
    }

    /// Calculates the Dagum distribution CDF at a given point x.
    /// 
    /// The parameters are in the form (A, B, P) and the CDF is calculated as
    /// (1 + (x / B)^(-A))^(-P).
    fn dagum(&self, x:Option<f64>) -> f64 {

        if x == None {
            return 1.0;
        }
        let x = x.unwrap();

        (1. + (x / self.parameters[1]).powf(-self.parameters[0])).powf(-self.parameters[2])
    }
}