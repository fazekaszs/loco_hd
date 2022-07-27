pub struct WeightFunction {
    pub function_name: String,
    pub parameters: Vec<f64>
}

impl WeightFunction {

    pub fn new(function_name: String, parameters: Vec<f64>) -> Self {

        let length_assert = |length: usize| {
            assert!(parameters.len() == length, 
                "For function \"{}\" there must be exactly {} parameters!", function_name, length);
        };

        match function_name.as_str() {
            
            "hyper_exp" => {
                assert!(parameters.len() % 2 == 0, 
                    "For function \"{}\" there must be an even number of parameters!", function_name);
                for &item in &parameters {
                    assert!(item > 0., 
                        "For function \"{}\" all parameters mut be positive!", function_name);
                }
            },

            "dagum" => {
                length_assert(3);
                assert!(parameters[0] > 0. && parameters[1] > 0. && parameters[2] > 0., 
                    "For function \"{}\" all parameters mut be positive!", function_name);
            },


            "uniform" => {
                length_assert(2);
                assert!(parameters[0] >= 0., 
                    "For function \"{}\" the first parameter mut be non-negative!", function_name);
                assert!(parameters[1] > 0., 
                    "For function \"{}\" the second parameter mut be positive!", function_name);
                assert!(parameters[0] < parameters[1], 
                    "For function \"{}\" the first parameter must be smaller than the second!", function_name);
            },

            "kumarasawamy" => {
                length_assert(4);
                assert!(parameters[0] >= 0., 
                    "For function \"{}\" the first parameter mut be non-negative!", function_name);
                assert!(parameters[1] > 0. && parameters[2] > 0. && parameters[3] > 0., 
                    "For function \"{}\" after the first parameter all parameters mut be positive!", function_name);
                assert!(parameters[0] < parameters[1], 
                    "For function \"{}\" the first parameter must be smaller than the second!", function_name);
            },

            other => unimplemented!("No function implemented with name \"{}\"!", other)
        }

        Self { function_name, parameters }
    }

    pub fn integral(&self, x: Option<f64>) -> f64 {
        match self.function_name.as_str() {
            "hyper_exp" => self.hyper_exp(x),
            "dagum" => self.dagum(x),
            "uniform" => self.uniform(x),
            "kumarasawamy" => self.kumarasawamy(x),
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
    fn dagum(&self, x: Option<f64>) -> f64 {

        if x == None {
            return 1.0;
        }
        let x = x.unwrap();

        (1. + (x / self.parameters[1]).powf(-self.parameters[0])).powf(-self.parameters[2])
    }

    /// Calculates the uniform distribution CDF at a given point x.
    /// 
    /// The parameters are in the form (x_min, x_max) and the CDF is calculated in the following
    /// way:
    /// 
    /// - 0, if x < x_min
    /// - (x - x_min) / (x_max - x_min), if x_min <= x <= x_max 
    /// - 1.0, if x > x_max.
    fn uniform(&self, x: Option<f64>) -> f64 {

        if x == None {
            return 1.0;
        }
        let x = x.unwrap();

        if x < self.parameters[0] {
            return 0.0;
        }
        if x > self.parameters[1] {
            return 1.0;
        }

        (x - self.parameters[0]) / (self.parameters[1] - self.parameters[0])
    }

    /// Calculates the Kumarasawamy distribution CDF at a given point x. It is generalized to an arbitrary
    /// interval by the parameters.
    /// 
    /// The parameters are in the form (x_min, x_max, A, B) and the CDF is calculated in the following
    /// way:
    /// 
    /// - 0, if x < x_min
    /// - y = (x - x_min) / (x_max - x_min), if x_min <= x <= x_max, and then 1 - (1 - y^A)^B
    /// - 1.0, if x > x_max.
    fn kumarasawamy(&self, x: Option<f64>) -> f64 {

        if x == None {
            return 1.0;
        }
        let mut x = x.unwrap();

        if x < self.parameters[0] {
            return 0.0;
        }
        if x > self.parameters[1] {
            return 1.0;
        }

        x = (x - self.parameters[0]) / (self.parameters[1] - self.parameters[0]);
        1. - (1. - x.powf(self.parameters[2])).powf(self.parameters[3])
    }

}