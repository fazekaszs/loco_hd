use super::weight_function::*;

use pyo3::prelude::*;

fn approx_eq(test_value: f64, goal_value: f64) {
    assert!(goal_value - 1E-4 < test_value && test_value < goal_value + 1E-4);
}

#[test]
fn build_hyper_exp_ok() -> PyResult<()> {

    let wf = WeightFunction::build(
        "hyper_exp".to_owned(), 
        vec![1., 1.]
    )?;

    let wf_val = wf.integral_range(0., 1.)?;
    approx_eq(wf_val, 0.6321);
    let wf_val = wf.integral_range(1., 3.)?;
    approx_eq(wf_val, 0.3180);
    let wf_val = wf.integral_range(5., 10.)?;
    approx_eq(wf_val, 0.0066);

    let wf = WeightFunction::build(
        "hyper_exp".to_owned(), 
        vec![0.5, 0.5, 1. / 2., 1. / 3.]
    )?;

    let wf_val = wf.integral_range(0., 1.)?;
    approx_eq(wf_val, 0.3385);
    let wf_val = wf.integral_range(1., 3.)?;
    approx_eq(wf_val, 0.3661);
    let wf_val = wf.integral_range(5., 10.)?;
    approx_eq(wf_val, 0.1143);

    let wf = WeightFunction::build(
        "hyper_exp".to_owned(), 
        vec![3., 5., 2., 1. / 3., 1. / 5., 1. / 10.]
    )?;

    let wf_val = wf.integral_range(0., 1.)?;
    approx_eq(wf_val, 0.1947);
    let wf_val = wf.integral_range(1., 3.)?;
    approx_eq(wf_val, 0.2723);
    let wf_val = wf.integral_range(5., 10.)?;
    approx_eq(wf_val, 0.2099);

    Ok(())
}

#[test]
fn build_hyper_exp_err() {

    let wf = WeightFunction::build(
        "hyper_exp".to_owned(), 
        vec![1., ]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "hyper_exp".to_owned(), 
        vec![1., 2., 3.]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "hyper_exp".to_owned(), 
        vec![-1., 1.]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "hyper_exp".to_owned(), 
        vec![1., -1.]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "hyper_exp".to_owned(), 
        vec![-1., 1., 2.]
    );
    assert!(wf.is_err());
}

#[test]
fn build_dagum_ok() -> PyResult<()> {

    let wf = WeightFunction::build(
        "dagum".to_owned(), 
        vec![1., 1., 1.]
    )?;

    let wf_val = wf.integral_range(0., 1.)?;
    approx_eq(wf_val, 0.5000);
    let wf_val = wf.integral_range(1., 3.)?;
    approx_eq(wf_val, 0.2500);
    let wf_val = wf.integral_range(5., 10.)?;
    approx_eq(wf_val, 0.0757);

    let wf = WeightFunction::build(
        "dagum".to_owned(), 
        vec![2., 5., 1.]
    )?;

    let wf_val = wf.integral_range(0., 1.)?;
    approx_eq(wf_val, 0.0384);
    let wf_val = wf.integral_range(1., 3.)?;
    approx_eq(wf_val, 0.2262);
    let wf_val = wf.integral_range(5., 10.)?;
    approx_eq(wf_val, 0.3000);

    let wf = WeightFunction::build(
        "dagum".to_owned(), 
        vec![10., 5., 2.]
    )?;

    let wf_val = wf.integral_range(0., 1.)?;
    approx_eq(wf_val, 0.0000);
    let wf_val = wf.integral_range(1., 3.)?;
    approx_eq(wf_val, 0.0000);
    let wf_val = wf.integral_range(5., 10.)?;
    approx_eq(wf_val, 0.7480);

    Ok(())
}

#[test]
fn build_dagum_err() {

    let wf = WeightFunction::build(
        "dagum".to_owned(), 
        vec![1., ]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "dagum".to_owned(), 
        vec![1., 2.]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "dagum".to_owned(), 
        vec![-1., 2., 3.]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "dagum".to_owned(), 
        vec![1., -2., 3.]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "dagum".to_owned(), 
        vec![1., 2., -3.]
    );
    assert!(wf.is_err());
}

#[test]
fn build_uniform_ok() -> PyResult<()> {

    let wf = WeightFunction::build(
        "uniform".to_owned(), 
        vec![0., 1.]
    )?;

    let wf_val = wf.integral_range(0., 1.)?;
    approx_eq(wf_val, 1.0000);
    let wf_val = wf.integral_range(1., 3.)?;
    approx_eq(wf_val, 0.0000);
    let wf_val = wf.integral_range(5., 10.)?;
    approx_eq(wf_val, 0.0000);

    let wf = WeightFunction::build(
        "uniform".to_owned(), 
        vec![3., 10.]
    )?;        

    let wf_val = wf.integral_range(0., 1.)?;
    approx_eq(wf_val, 0.0000);
    let wf_val = wf.integral_range(1., 3.)?;
    approx_eq(wf_val, 0.0000);
    let wf_val = wf.integral_range(5., 10.)?;
    approx_eq(wf_val, 0.7142);

    let wf = WeightFunction::build(
        "uniform".to_owned(), 
        vec![2., 16.]
    )?;        

    let wf_val = wf.integral_range(0., 1.)?;
    approx_eq(wf_val, 0.0000);
    let wf_val = wf.integral_range(1., 3.)?;
    approx_eq(wf_val, 0.0714);
    let wf_val = wf.integral_range(5., 10.)?;
    approx_eq(wf_val, 0.3571);

    Ok(())
}

#[test]
fn build_uniform_err() {

    let wf = WeightFunction::build(
        "uniform".to_owned(), 
        vec![1., ]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "uniform".to_owned(), 
        vec![1., 0.]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "uniform".to_owned(), 
        vec![-1., 0.]
    );
    assert!(wf.is_err());
}

#[test]
fn build_kumaraswamy_ok() -> PyResult<()> {

    let wf = WeightFunction::build(
        "kumaraswamy".to_owned(), 
        vec![1., 2., 2., 2.]
    )?;

    let wf_val = wf.integral_range(1., 2.)?;
    approx_eq(wf_val, 1.0000);
    let wf_val = wf.integral_range(1.25, 1.75)?;
    approx_eq(wf_val, 0.6875);
    let wf_val = wf.integral_range(1.4, 10.)?;
    approx_eq(wf_val, 0.7056);

    let wf = WeightFunction::build(
        "kumaraswamy".to_owned(), 
        vec![5., 10., 2., 3.]
    )?;

    let wf_val = wf.integral_range(5., 7.)?;
    approx_eq(wf_val, 0.4072);
    let wf_val = wf.integral_range(1., 17.)?;
    approx_eq(wf_val, 1.0000);
    let wf_val = wf.integral_range(6.4, 6.7)?;
    approx_eq(wf_val, 0.0910);

    let wf = WeightFunction::build(
        "kumaraswamy".to_owned(), 
        vec![5., 9., 7., 7.]
    )?;

    let wf_val = wf.integral_range(5., 7.)?;
    approx_eq(wf_val, 0.0534);
    let wf_val = wf.integral_range(1., 17.)?;
    approx_eq(wf_val, 1.0000);
    let wf_val = wf.integral_range(6.4, 6.7)?;
    approx_eq(wf_val, 0.0129);

    Ok(())
}

#[test]
fn build_kumaraswamy_err() {

    let wf = WeightFunction::build(
        "kumaraswamy".to_owned(), 
        vec![1., ]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "kumaraswamy".to_owned(), 
        vec![1., 2.]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "kumaraswamy".to_owned(), 
        vec![3., 1., 2., 2.]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "kumaraswamy".to_owned(), 
        vec![1., 3., -2., 2.]
    );
    assert!(wf.is_err());

    let wf = WeightFunction::build(
        "kumaraswamy".to_owned(), 
        vec![0., 3., 2., -2.]
    );
    assert!(wf.is_err());
}