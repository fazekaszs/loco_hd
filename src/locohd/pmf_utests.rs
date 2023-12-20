use super::pmf::*;

use pyo3::prelude::*;

fn approx_eq(test_value: f64, goal_value: f64) {
    assert!(goal_value - 1E-4 < test_value && test_value < goal_value + 1E-4);
}

#[test]
fn hdist_values() -> PyResult<()> {

    let cats = vec!["A".to_owned(), "B".to_owned(), "C".to_owned()]
        .into_iter()
        .enumerate()
        .map(|(a, b)| (b, a))
        .collect();
    let mut pmf_system = PMFSystem::new(&cats);

    pmf_system.update_pmf1(&"A".to_owned())?;
    pmf_system.update_pmf2(&"B".to_owned())?;

    approx_eq(pmf_system.hellinger_dist()?, 1.);

    pmf_system.update_pmf1(&"B".to_owned())?;
    pmf_system.update_pmf2(&"A".to_owned())?;

    approx_eq(pmf_system.hellinger_dist()?, 0.);

    pmf_system.update_pmf1(&"A".to_owned())?;

    approx_eq(pmf_system.hellinger_dist()?, 0.1200);

    pmf_system.update_pmf1(&"C".to_owned())?;

    approx_eq(pmf_system.hellinger_dist()?, 0.3826);

    Ok(())
}

#[test]
fn evaluate_pmf_system_err() {

    let cats = vec!["A".to_owned(), "B".to_owned(), "C".to_owned()]
        .into_iter()
        .enumerate()
        .map(|(a, b)| (b, a))
        .collect();
    let mut pmf_system = PMFSystem::new(&cats);

    assert!(pmf_system.hellinger_dist().is_err());
    assert!(pmf_system.update_pmf1(&"D".to_owned()).is_err());
    assert!(pmf_system.update_pmf2(&"D".to_owned()).is_err());

}