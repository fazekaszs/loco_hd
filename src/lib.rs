mod locohd;
pub use locohd::{WeightFunction, PrimitiveAtom, LoCoHD, TagPairingRule};

use pyo3::prelude::*;

#[pymodule]
fn loco_hd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WeightFunction>()?;
    m.add_class::<PrimitiveAtom>()?;
    m.add_class::<TagPairingRule>()?;
    m.add_class::<LoCoHD>()?;
    Ok(())
}
