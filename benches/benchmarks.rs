#![cfg(feature = "bench")]

use criterion::{criterion_group, criterion_main, Criterion};

use mlsnd::testing::{fake_points, make_rng, read_cps, ref_points};
use mlsnd::{Float, PointMLS};
use moving_least_squares::deform_affine;

fn crate_deform_affine_multi<T: Float, const D: usize>(
    mls: &PointMLS<T, D>,
    points: &[[T; D]],
) -> Vec<[T; D]> {
    points.iter().map(|p| mls.transform(*p)).collect()
}

pub fn mlsnd_2d(c: &mut Criterion) {
    let (c_p, c_q) = read_cps::<2>();
    let mut rng = make_rng();
    let orig = fake_points(&c_p, 1000, &mut rng);

    let mls = PointMLS::new(c_p, c_q).expect("Could not construct");

    c.bench_function("mlsnd 2d f32", |b| {
        b.iter(|| crate_deform_affine_multi(&mls, &orig))
    });
}

pub fn mlsnd_3d_f32(c: &mut Criterion) {
    let (c_p, c_q) = read_cps::<3>();
    let mut rng = make_rng();
    let orig = fake_points(&c_p, 1000, &mut rng);

    let mls = PointMLS::new(c_p, c_q).expect("Could not construct");

    c.bench_function("mlsnd 3D f32", |b| {
        b.iter(|| crate_deform_affine_multi(&mls, &orig))
    });
}

fn f32_to_64<const D: usize>(v: &[[f32; D]]) -> Vec<[f64; D]> {
    v.iter()
        .map(|p| {
            p.iter()
                .map(|n| *n as f64 + f64::EPSILON)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect()
}

pub fn mlsnd_3d_f64(c: &mut Criterion) {
    let (c_p, c_q) = read_cps::<3>();
    let mut rng = make_rng();
    let orig = fake_points(&c_p, 1000, &mut rng);

    let mls = PointMLS::new(f32_to_64(&c_p), f32_to_64(&c_q)).expect("Could not construct");

    let orig64 = f32_to_64(&orig);

    c.bench_function("mlsnd 3D f64", |b| {
        b.iter(|| crate_deform_affine_multi(&mls, &orig64))
    });
}

fn ref_deform_affine_multi(
    controls_p: &[(f32, f32)],
    controls_q: &[(f32, f32)],
    points: &[(f32, f32)],
) -> Vec<(f32, f32)> {
    points
        .iter()
        .map(|p| deform_affine(controls_p, controls_q, *p))
        .collect()
}

pub fn moving_least_squares_2d(c: &mut Criterion) {
    let (c_p, c_q) = read_cps::<2>();
    let mut rng = make_rng();
    let orig = ref_points(&fake_points(&c_p, 1000, &mut rng));

    let cont_p = ref_points(&c_p);
    let cont_q = ref_points(&c_q);

    c.bench_function("moving-least-squares 2D f32", |b| {
        b.iter(|| ref_deform_affine_multi(&cont_p, &cont_q, &orig))
    });
}

criterion_group!(comparison_2d, mlsnd_2d, moving_least_squares_2d);
criterion_group!(alone_3d, mlsnd_3d_f32, mlsnd_3d_f64);
criterion_main!(comparison_2d, alone_3d);
