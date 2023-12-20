/// Calculates the hyper-exponential distribution CDF at a given point x.
/// 
/// The parameters are in the form of (A_1, A_2, ... A_n, B_1, B_2, ... B_n) and the
/// CDF is calculated as 1 - SUM_i(A_i * exp(-B_i * x)) / SUM_i(A_i).
pub fn hyper_exp(parameters: &Vec<f64>, x: f64) -> f64 {

    let mut norm = 0f64;
    let mut sum = 0f64;

    let (params_a, params_b) = {
        let border_idx = parameters.len() / 2;
        (&parameters[..border_idx], &parameters[border_idx..])
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
pub fn dagum(parameters: &Vec<f64>, x: f64) -> f64 {
    (1. + (x / parameters[1]).powf(-parameters[0])).powf(-parameters[2])
}

/// Calculates the uniform distribution CDF at a given point x.
/// 
/// The parameters are in the form (x_min, x_max) and the CDF is calculated in the following
/// way:
/// 
/// - 0, if x < x_min
/// - (x - x_min) / (x_max - x_min), if x_min <= x <= x_max 
/// - 1.0, if x > x_max.
pub fn uniform(parameters: &Vec<f64>, x: f64) -> f64 {

    if x < parameters[0] { return 0f64; }
    if x > parameters[1] { return 1f64 }

    (x - parameters[0]) / (parameters[1] - parameters[0])
}

/// Calculates the Kumaraswamy distribution CDF at a given point x. It is generalized to an arbitrary
/// interval by the parameters.
/// 
/// The parameters are in the form (x_min, x_max, A, B) and the CDF is calculated in the following
/// way:
/// 
/// - 0, if x < x_min
/// - y = (x - x_min) / (x_max - x_min), if x_min <= x <= x_max, and then 1 - (1 - y^A)^B
/// - 1.0, if x > x_max.
pub fn kumaraswamy(parameters: &Vec<f64>, x: f64) -> f64 {

    if x < parameters[0] { return 0f64; }
    if x > parameters[1] { return 1f64; }

    let z = (x - parameters[0]) / (parameters[1] - parameters[0]);
    1. - (1. - z.powf(parameters[2])).powf(parameters[3])
}