mod cdfs;

use pyo3::{prelude::*, exceptions::PyValueError};

#[pyclass]
#[derive(Clone)]
pub struct WeightFunction {

    #[pyo3(get)]
    pub parameters: Vec<f64>,

    #[pyo3(get)]
    pub function_name: String,

    pub function: fn(&Vec<f64>, f64) -> f64,
}

#[pymethods]
impl WeightFunction {

    #[new]
    pub fn build(function_name: String, parameters: Vec<f64>) -> PyResult<Self> {

        let length_assert = |length: usize| {
            let err_msg = format!("For function \"{}\" there must be exactly {} parameters!", function_name, length);
            if parameters.len() != length { Err(PyValueError::new_err(err_msg)) } else { Ok(()) }
        };

        let function = match function_name.as_str() {

            "hyper_exp" => {

                let err_msg = format!("For function \"{}\" there must be an even number of parameters!", function_name);
                if parameters.len() % 2 != 0 { return Err(PyValueError::new_err(err_msg)) };

                let err_msg = format!("For function \"{}\" all parameters must be positive!", function_name);
                for &item in &parameters { if item <= 0f64 { return Err(PyValueError::new_err(err_msg)); }}

                cdfs::hyper_exp
            },

            "dagum" => {

                length_assert(3)?;

                let err_msg = format!("For function \"{}\" all parameters must be positive!", function_name);
                if parameters[0] < 0f64 || parameters[1] < 0f64 || parameters[2] < 0f64 { return Err(PyValueError::new_err(err_msg)); }

                cdfs::dagum
            },


            "uniform" => {

                length_assert(2)?;

                let err_msg = format!("For function \"{}\" the first parameter must be non-negative!", function_name);
                if parameters[0] < 0f64 { return Err(PyValueError::new_err(err_msg)); }

                let err_msg = format!("For function \"{}\" the second parameter must be positive!", function_name);
                if parameters[1] <= 0f64 { return Err(PyValueError::new_err(err_msg)); }

                let err_msg = format!("For function \"{}\" the first parameter must be smaller than the second!", function_name);
                if parameters[0] >= parameters[1] { return Err(PyValueError::new_err(err_msg)); }

                cdfs::uniform
            },

            "kumaraswamy" => {

                length_assert(4)?;

                let err_msg = format!("For function \"{}\" the first parameter must be non-negative!", function_name);
                if parameters[0] < 0f64 { return Err(PyValueError::new_err(err_msg)); }

                let err_msg = format!("For function \"{}\" after the first parameter all parameters must be positive!", function_name);
                if parameters[1] <= 0f64 || parameters[2] <= 0f64 || parameters[3] <= 0f64 { return Err(PyValueError::new_err(err_msg)); }

                let err_msg = format!("For function \"{}\" the first parameter must be smaller than the second!", function_name);
                if parameters[0] >= parameters[1] { return Err(PyValueError::new_err(err_msg)); }

                cdfs::kumaraswamy
            },

            other => {

                let err_msg = format!("No function implemented with name \"{}\"!", other);
                return Err(PyValueError::new_err(err_msg));
            }
        };

        Ok(Self { parameters, function_name, function })
    }

    pub fn integral_point(&self, point: f64) -> PyResult<f64> {

        if point < 0. {
            let err_msg = format!("Invalid input value: {}. All values must be non-negative!", point);
            return Err(PyValueError::new_err(err_msg));
        }

        Ok((self.function)(&self.parameters, point))
    }

    pub fn integral_vec(&self, points: Vec<f64>) -> PyResult<Vec<f64>> {

        let mut output = Vec::new();
        for x in points {
            match self.integral_point(x) {
                Ok(result) => output.push(result),
                Err(err) => return Err(err)
            }
        }

        Ok(output)
    }

    pub fn integral_range(&self, point_from: f64, point_to: f64) -> PyResult<f64> {
        Ok(self.integral_point(point_to)? - self.integral_point(point_from)?)
    }
}
