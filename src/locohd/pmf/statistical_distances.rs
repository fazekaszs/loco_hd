use pyo3::{pyclass, pymethods, PyResult};
use pyo3::exceptions::PyValueError;

fn hellinger_distance(p1: &[f64], p2: &[f64], params: &[f64]) -> PyResult<f64> {
    let exponent = params[0];
    let sum_core = |(x, y): (&f64, &f64)| {
        (x.powf(1. / exponent) - y.powf(1. / exponent)).abs().powf(exponent)
    };
    let dist = p1.iter().zip(p2).map(sum_core).sum::<f64>();
    Ok((dist / 2f64).powf(1. / exponent))
}

fn kolmogorov_smirnov_distance(p1: &[f64], p2: &[f64], _params: &[f64]) -> PyResult<f64> {
    let sum_core = |(x, y): (&f64, &f64)| { (x - y).abs() };
    let dist = p1
        .iter()
        .zip(p2)
        .map(sum_core)
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    Ok(dist)
}

#[pyclass]
#[derive(Clone)]
pub struct StatisticalDistance {
    raw_function: fn(&[f64], &[f64], &[f64]) -> PyResult<f64>,
    parameters: Vec<f64>
}

#[pymethods]
impl StatisticalDistance {
    #[new]
    pub fn build(distance_name: String, parameters: Vec<f64>) -> PyResult<Self> {

        // TODO: check for parameter validity!!
        let raw_function = match distance_name.as_str() {

            "Hellinger" => { 
                hellinger_distance
            },
            "Kolmogorov-Smirnov" => { 
                kolmogorov_smirnov_distance 
            },
            other => {
                let err_msg = format!("Invalid statistical distance name {}!", other);
                return Err(PyValueError::new_err(err_msg))
            }
        };
        Ok(Self { raw_function, parameters })
    }
    
    pub fn run(&self, p1: Vec<f64>, p2: Vec<f64>) -> PyResult<f64> {
        (self.raw_function)(&p1, &p2, &self.parameters)
    }
    
}

