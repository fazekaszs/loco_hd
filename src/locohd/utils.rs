pub fn euclidean_distance(vec_a: [f64; 3], vec_b: [f64; 3]) -> f64 {
            
    let mut distance = 0.;
    for (item_a, item_b) in vec_a.into_iter().zip(vec_b) {
        distance += (item_a - item_b).powf(2.);
    }
    distance.powf(0.5)
}

pub fn calculate_distance_matrix(coords: &Vec<[f64; 3]>) -> Vec<Vec<f64>> {

    let mut distance_mx = vec![vec![0.; coords.len()]; coords.len()];
            
    for idx1 in 0..coords.len() {
        for idx2 in idx1 + 1..coords.len() {
            let distance = euclidean_distance(coords[idx1], coords[idx2]);
            distance_mx[idx1][idx2] = distance;
            distance_mx[idx2][idx1] = distance;
        }
    }
    distance_mx
}

// The sort_together function co-sorts a distance matrix line with a list of categories.
pub fn sort_together(dists: &Vec<f64>, cats: &Vec<String>) -> (Vec<f64>, Vec<String>) {

    let mut mask = (0..dists.len()).collect::<Vec<_>>();
    mask.sort_by(|&idx1, &idx2| dists[idx1].partial_cmp(&dists[idx2]).unwrap());

    let mut new_dmx_line = vec![];
    let mut new_cat = vec![];

    for &idx in mask.iter() {
        new_dmx_line.push(dists[idx]);
        new_cat.push(cats[idx].clone());
    }

    (new_dmx_line, new_cat)
    
}