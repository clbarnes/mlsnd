use moving_least_squares::deform_affine;
use std::fmt::Debug;
use std::fs;
use std::iter::repeat_with;
use std::path::PathBuf;

use crate::Precision;

fn data_dir() -> PathBuf {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("data");
    d
}

fn read_cp_rows() -> Vec<Vec<Precision>> {
    let mut cp_path = data_dir();
    cp_path.push("control_points.csv");
    let contents = &fs::read_to_string(cp_path).expect("Couldn't read file")[..];

    let mut out = Vec::default();

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let elements: Vec<_> = trimmed
            .split(",")
            .map(|e| e.parse().expect("Could not parse float"))
            .collect();
        out.push(elements);
    }

    out
}

pub fn read_cps<const D: usize>() -> (Vec<[Precision; D]>, Vec<[Precision; D]>) {
    if D > 3 {
        panic!("Maximum dimensionality is 3");
    }
    let mut cp1 = Vec::default();
    let mut cp2 = Vec::default();

    for row in read_cp_rows().into_iter() {
        cp1.push(
            row[0..D]
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );
        cp2.push(
            row[3..3 + D]
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );
    }

    (cp1, cp2)
}

/// Find a random point between the two given points.
fn mean_point<const D: usize>(p1: &[f32; D], p2: &[f32; D], rng: &mut fastrand::Rng) -> [f32; D] {
    p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| {
            let r = rng.f32();
            (a * r + b * (1.0 - r)) / 2.0
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Randomly pick pairs of points from a set of points (e.g. control points)
/// and randomly create a new point on the line segment between them.
///
/// This ensures that the points are within the field of control points, but are random.
pub fn fake_points<const D: usize>(
    field: &[[f32; D]],
    n: usize,
    rng: &mut fastrand::Rng,
) -> Vec<[f32; D]> {
    repeat_with(|| {
        let p1 = field[rng.usize(0..field.len())];
        let p2 = field[rng.usize(0..field.len())];
        mean_point(&p1, &p2, rng)
    })
    .take(n)
    .collect()
}

pub fn make_rng() -> fastrand::Rng {
    fastrand::Rng::with_seed(1991)
}

fn ref_point(p: &[f32; 2]) -> (f32, f32) {
    (p[0], p[1])
}

pub fn ref_points(ps: &[[f32; 2]]) -> Vec<(f32, f32)> {
    ps.iter().map(ref_point).collect()
}

pub fn ref_deform(
    controls_p: &[[f32; 2]],
    controls_q: &[[f32; 2]],
    points: &[[f32; 2]],
) -> Vec<[f32; 2]> {
    let c_p = ref_points(controls_p);
    let c_q = ref_points(controls_q);

    points
        .iter()
        .map(|p| {
            let p2 = deform_affine(&c_p, &c_q, ref_point(p));
            [p2.0, p2.1]
        })
        .collect()
}

struct WrapPt<const D: usize, T: Sized + Debug + PartialEq>([T; D]);

impl<const D: usize, T: Sized + Debug + PartialEq> PartialEq for WrapPt<D, T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

fn assert_eq_pt<const D: usize>(p1: &[f32; D], p2: &[f32; D], epsilon: f32) {
    if p1
        .iter()
        .zip(p2.iter())
        .all(|(a, b)| (a - b).abs() < epsilon)
    {
        return;
    }
    panic!(
        "Points have unequal elements (epsilon = {}).\n  left: {:?}\n right: {:?}",
        epsilon, p1, p2
    );
}

pub fn assert_eq_sl(v1: &[[f32; 2]], v2: &[[f32; 2]]) {
    if v1.len() != v2.len() {
        panic!("Vecs have different lengths");
    }
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert_eq_pt(a, b, 0.1)
    }
}
