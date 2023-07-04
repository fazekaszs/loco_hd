use super::*;
use super::tag_pairing_rule::TagPairingRuleVariants;

fn approx_eq(test_value: f64, goal_value: f64) {
    assert!(goal_value - 1E-4 < test_value && test_value < goal_value + 1E-4);
}

#[test]
fn value_test1() -> PyResult<()> {

    let w_func = WeightFunction::build(
        "uniform".to_owned(), 
        vec![0., 4.]
    )?;
    let cats = vec!["O".to_owned(), "A".to_owned(), "B".to_owned(), "C".to_owned()];
    let tag_pairing_rule = TagPairingRule::build(
        TagPairingRuleVariants::WithoutList { accept_same: false }
    )?;
    let lchd = LoCoHD::build(cats, w_func, tag_pairing_rule, None)?;

    let result = lchd.from_anchors(
        vec!["O".to_owned(), "A".to_owned(), "B".to_owned(), "C".to_owned()], 
        vec!["O".to_owned(), "A".to_owned(), "B".to_owned(), "C".to_owned()],
        vec![0., 1., 2., 3.],
        vec![0., 1., 1., 1.]
    )?;

    approx_eq(result, 0.2268);

    let result = lchd.from_anchors(
        vec!["O".to_owned(), "A".to_owned(), "B".to_owned(), "C".to_owned()], 
        vec!["O".to_owned(), "A".to_owned(), "B".to_owned(), "C".to_owned()],
        vec![0., 1., 1., 1.],
        vec![0., 1., 2., 3.]
    )?;

    approx_eq(result, 0.2268);

    Ok(())
}