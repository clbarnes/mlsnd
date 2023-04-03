#![cfg(feature = "bench")]

use criterion::{criterion_group, criterion_main, Criterion};

use mlsnd::testing::{fake_points, make_rng, read_cps, ref_points};
use mlsnd::PointMLS;
use moving_least_squares::deform_affine;

fn crate_deform_affine_multi<const D: usize>(
    mls: &PointMLS<D>,
    points: &[[f32; D]],
) -> Vec<[f32; D]> {
    points.iter().map(|p| mls.transform(*p)).collect()
}

pub fn mlsnd_2d(c: &mut Criterion) {
    let (c_p, c_q) = read_cps::<2>();
    let mut rng = make_rng();
    let orig = fake_points(&c_p, 100, &mut rng);

    let mls = PointMLS::new(c_p, c_q).expect("Could not construct");

    c.bench_function("mlsnd_2d", |b| {
        b.iter(|| crate_deform_affine_multi(&mls, &orig))
    });
}

pub fn mlsnd_3d(c: &mut Criterion) {
    let (c_p, c_q) = read_cps::<3>();
    let mut rng = make_rng();
    let orig = fake_points(&c_p, 100, &mut rng);

    let mls = PointMLS::new(c_p, c_q).expect("Could not construct");

    c.bench_function("mlsnd_3d", |b| {
        b.iter(|| crate_deform_affine_multi(&mls, &orig))
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
    let orig = ref_points(&fake_points(&c_p, 100, &mut rng));

    let cont_p = ref_points(&c_p);
    let cont_q = ref_points(&c_q);

    c.bench_function("moving_least_squares_2d", |b| {
        b.iter(|| ref_deform_affine_multi(&cont_p, &cont_q, &orig))
    });
}

criterion_group!(comparison_2d, mlsnd_2d, moving_least_squares_2d);
criterion_group!(alone_3d, mlsnd_2d, mlsnd_3d);
criterion_main!(comparison_2d, alone_3d);
