use ndarray::prelude::*;
use ndarray::Data;

use std::cmp::Ordering;

// argsort_by function from: https://github.com/rust-ndarray/ndarray/issues/1145
pub fn argsort_by<S, F>(arr: &ArrayBase<S, Ix1>, mut compare: F) -> Vec<usize>
where
    S: Data,
    F: FnMut(&S::Elem, &S::Elem) -> Ordering,
{
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_by(move |&i, &j| compare(&arr[i], &arr[j]));
    indices
}

pub(crate) fn vec_to_bbox<T: Copy>(v: Vec<T>) -> [T; 4] {
    return [v[0], v[1], v[2], v[3]];
}

#[cfg(feature = "save")]
pub(crate) mod save {
    use ndarray::{Array1, Array2};

    pub fn savetxt(a: &Array2<f32>, filename: &str) {
        let file = std::fs::File::create(filename).unwrap();
        let mut writer = csv::Writer::from_writer(file);
        for row in a.outer_iter() {
            writer.serialize(row.iter().collect::<Vec<_>>()).unwrap();
        }
        writer.flush().unwrap();
    }

    pub fn savetxt_u32(a: &Array2<u32>, filename: &str) {
        let file = std::fs::File::create(filename).unwrap();
        let mut writer = csv::Writer::from_writer(file);
        for row in a.outer_iter() {
            writer.serialize(row.iter().collect::<Vec<_>>()).unwrap();
        }
        writer.flush().unwrap();
    }

    pub fn savetxt_usize(a: &Array2<usize>, filename: &str) {
        let file = std::fs::File::create(filename).unwrap();
        let mut writer = csv::Writer::from_writer(file);
        for row in a.outer_iter() {
            writer.serialize(row.iter().collect::<Vec<_>>()).unwrap();
        }
        writer.flush().unwrap();
    }

    pub fn savetxt_usize_a1(a: &Array1<usize>, filename: &str) {
        let file = std::fs::File::create(filename).unwrap();
        let mut writer = csv::Writer::from_writer(file);
        for row in a.outer_iter() {
            writer.serialize(row.iter().collect::<Vec<_>>()).unwrap();
        }
        writer.flush().unwrap();
    }

    pub fn savetxt_f32_a1(a: &Array1<f32>, filename: &str) {
        let file = std::fs::File::create(filename).unwrap();
        let mut writer = csv::Writer::from_writer(file);
        for row in a.outer_iter() {
            writer.serialize(row.iter().collect::<Vec<_>>()).unwrap();
        }
        writer.flush().unwrap();
    }
}
