mod locohd;
pub use locohd::{WeightFunction, PrimitiveAtom, LoCoHD};

use pyo3::prelude::*;

#[pymodule]
fn loco_hd(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WeightFunction>()?;
    m.add_class::<PrimitiveAtom>()?;
    m.add_class::<LoCoHD>()?;
    Ok(())
}
