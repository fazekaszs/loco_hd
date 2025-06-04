use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

fn hellinger_distance(p1: &[f64], p2: &[f64], exponent: f64) -> PyResult<f64> {
    let map_core = |(x, y): (&f64, &f64)| {
        (x.powf(1. / exponent) - y.powf(1. / exponent)).abs().powf(exponent)
    };
    let dist = p1.iter().zip(p2).map(map_core).sum::<f64>();
    Ok((dist / 2f64).powf(1. / exponent))
}

fn kolmogorov_smirnov_distance(p1: &[f64], p2: &[f64]) -> PyResult<f64> {
    let map_core = |(x, y): (&f64, &f64)| { (x - y).abs() };
    let dist = p1
        .iter()
        .zip(p2)
        .map(map_core)
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    Ok(dist)
}

fn kullback_leibler_divergence(p1: &[f64], p2: &[f64], epsilon: f64) -> PyResult<f64> {
    let map_core = |(x, y): (&f64, &f64)| {
        x * ((x + epsilon) / (y + epsilon)).ln()
    };
    let dist = p1.iter().zip(p2).map(map_core).sum::<f64>();
    Ok(dist)
}

fn renyi_divergence(p1: &[f64], p2: &[f64], alpha: f64, epsilon: f64) -> PyResult<f64> {
    match alpha {

        // Special case alpha = 1:
        // It is a limit to the KL divergence.
        1. => {
            kullback_leibler_divergence(p1, p2, epsilon)
        },

        // Special case alpha = inf:
        // It is a limit to the log of the max probability ratio.
        f64::INFINITY => {
            let result = p1
                .iter()
                .zip(p2)
                .map(|(&x, &y)| (x + epsilon) / (y + epsilon))
                .max_by(|&r1, &r2| r1.partial_cmp(&r2).unwrap())
                .unwrap()
                .ln();
            Ok(result)
        },

        // Special case alpha = 0:
        // It is a limit to the minus log (sum p2(i) for i where p1(i) > 0).
        0. => {
            let result = p1
                .iter()
                .zip(p2)
                .filter(|(&x, _)| x > 0f64)
                .map(|(_, &y)| y)
                .sum::<f64>()
                .ln();
            Ok(-result)
        },

        // All other cases are computable by the classical formula.
        _ => {
            let result = p1
                .iter()
                .zip(p2)
                .map(
                    |(&x, &y)| x * ((x + epsilon) / (y + epsilon)).powf(alpha - 1.)
                ).sum::<f64>()
                .ln();
            Ok(result / (alpha - 1.))
        }
    }
}

#[derive(Clone)]
enum StatDistName {
    Hellinger([f64; 1]),
    KolmogorovSmirnov([f64; 0]),
    KullbackLeibler([f64; 1]),
    Renyi([f64; 2])
}

#[pyclass]
#[derive(Clone)]
pub struct StatisticalDistance{
    distance_name: StatDistName
}

#[pymethods]
impl StatisticalDistance {
    #[new]
    pub fn build(distance_name: String, parameters: Vec<f64>) -> PyResult<Self> {

        let n_params_error = |v: Vec<f64>| {
            let err_msg = format!(
                "Invalid number of parameters for {}: {}", distance_name, v.len()
            );
            PyValueError::new_err(err_msg)
        };

        let distance_name = match distance_name.as_str() {
            "Hellinger" => 
                StatDistName::Hellinger(parameters.try_into().map_err(n_params_error)?),
            "Kolmogorov-Smirnov" => 
                StatDistName::KolmogorovSmirnov(parameters.try_into().map_err(n_params_error)?),
            "Kullback-Leibler" => 
                StatDistName::KullbackLeibler(parameters.try_into().map_err(n_params_error)?),
            "Renyi" => 
                StatDistName::Renyi(parameters.try_into().map_err(n_params_error)?),
            other => {
                let err_msg = format!("Invalid statistical distance name {}!", other);
                return Err(PyValueError::new_err(err_msg))
            }
        };
        Ok(Self { distance_name })
    }
    
    pub fn run(&self, p1: Vec<f64>, p2: Vec<f64>) -> PyResult<f64> {
        match &self.distance_name {

            StatDistName::Hellinger([exponent]) => {
                hellinger_distance(&p1, &p2, *exponent)
            },

            StatDistName::KolmogorovSmirnov([]) => {
                kolmogorov_smirnov_distance(&p1, &p2)
            },

            StatDistName::KullbackLeibler([epsilon]) => {
                kullback_leibler_divergence(&p1, &p2, *epsilon)
            },

            StatDistName::Renyi([alpha, epsilon]) => {
                renyi_divergence(&p1, &p2, *alpha, *epsilon)
            }
        }
    }
}

