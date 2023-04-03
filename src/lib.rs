pub use nalgebra;
use nalgebra::{distance_squared, Point, RealField, SMatrix, SVector};
use std::fmt::Debug;
use thiserror::Error;

#[cfg(any(test, feature = "bench"))]
pub mod testing;

pub trait Float: num_traits::Float + Debug + RealField {}

impl Float for f32 {}

impl Float for f64 {}

// todo: dynamic dimensions, otherwise for python/wasm you need to compile every dimension explicitly
// that said, explicit dimensions aren't verbose (they just might take up space)
// pub type PointMLS2 = PointMLS<2>;
// pub type PointMLS3 = PointMLS<3>;
// pub type PointMLS4 = PointMLS<4>;

struct Variables<T: Float, const D: usize> {
    pub w_all: Vec<T>,
    pub p_star: Point<T, D>,
    pub q_star: Point<T, D>,
    pub p_hat: Vec<Point<T, D>>,
}

enum VarOrPoint<T: Float, const D: usize> {
    Var(Variables<T, D>),
    Point(Point<T, D>),
}

impl<T: Float, const D: usize> VarOrPoint<T, D> {
    pub fn new(controls_p: &[Point<T, D>], controls_q: &[Point<T, D>], point: Point<T, D>) -> Self {
        let sqr_dist = |p| distance_squared(p, &point);
        let weight = |p| T::one() / sqr_dist(p);
        let mut w_sum = T::zero();
        let mut w_all = Vec::with_capacity(controls_p.len());

        let mut wp_star_sum = SVector::<_, D>::from_element(T::zero());
        let mut wq_star_sum = SVector::<_, D>::from_element(T::zero());

        for (idx, (p, q)) in controls_p.iter().zip(controls_q.iter()).enumerate() {
            let w = weight(p);
            if w.is_infinite() {
                // point of interest is on top of control point
                return VarOrPoint::Point(controls_q[idx]);
            }
            w_all.push(w);
            w_sum += w;

            wp_star_sum += p.coords * w;
            wq_star_sum += q.coords * w;
        }

        let p_star = wp_star_sum / w_sum;
        let q_star = wq_star_sum / w_sum;

        // // centroid p*
        // let wp_star_sum = w_all
        //     .iter()
        //     .zip(controls_p)
        //     .map(|(w, p)| (p * *w))
        //     .reduce(|acc, e| acc + e.coords)
        //     .expect("No control points");
        // let p_star = (1.0 / w_sum) * wp_star_sum;

        // // centroid q*
        // let wq_star_sum = w_all
        //     .iter()
        //     .zip(controls_q)
        //     .map(|(w, p)| (p * *w))
        //     .reduce(sum_p)
        //     .expect("No control points");
        // let q_star = (1.0 / w_sum) * wq_star_sum;

        // affine matrix M
        let p_hat: Vec<_> = controls_p.iter().map(|p| p - p_star).collect();

        Self::Var(Variables {
            w_all,
            p_star: p_star.into(),
            q_star: q_star.into(),
            p_hat,
        })
    }
}

pub fn deform_affine<T: Float, const D: usize>(
    controls_p: &[Point<T, D>],
    controls_q: &[Point<T, D>],
    point: Point<T, D>,
) -> Point<T, D> {
    let vars = match VarOrPoint::new(controls_p, controls_q, point) {
        VarOrPoint::Var(v) => v,
        VarOrPoint::Point(p) => return p,
    };

    let mut mp = SMatrix::<_, D, D>::from_element(T::zero());
    let mut mq = SMatrix::<_, D, D>::from_element(T::zero());

    for ((w, p_h), q) in vars.w_all.iter().zip(vars.p_hat.iter()).zip(controls_q) {
        mp += &(times_transpose(&p_h.coords, &p_h.coords) * *w);

        let qh = q - vars.q_star.coords;
        mq += &times_transpose(&(p_h.coords * *w), &qh.coords);
    }

    // // affine matrix M
    // let mp = vars.w_all
    //     .iter()
    //     .zip(&vars.p_hat)
    //     .map(|(w, p)| times_transpose(&p.coords, &p.coords) * *w)
    //     .reduce(sum_m)
    //     .expect("No control points");

    // let mq = vars.w_all
    //     .iter()
    //     .zip(&vars.p_hat)
    //     .zip(controls_q)
    //     .map(|((w, ph), q)| {
    //         let qh = q - vars.q_star.coords;
    //         // todo: check correct one is transposed `times_transpose`
    //         // pretty sure this = outer product, and correct
    //         times_transpose(&(ph * *w).coords, &qh.coords)
    //         // (ph * *w) * qh.coords.transpose()
    //     })
    //     .reduce(|acc, e| acc + e)
    //     .expect("No control points");

    let mp_inv = mp.try_inverse().expect("Matrix not invertible");

    transpose_mul(transpose_mul(point - vars.p_star.coords, mp_inv), mq) + vars.q_star.coords
}

#[derive(Debug, Copy, Clone)]
pub enum MLSStrategy {
    Affine,
    // other strategies are only covered in 2D in the paper
}

#[derive(Error, Debug)]
pub enum ConstructionError {
    #[error("Control point vectors have different lengths ({0}, {1})")]
    MismatchedControlPoints(usize, usize),
    #[error("No control points given")]
    NoControlPoints,
}

#[derive(Debug, Clone)]
pub struct PointMLS<T: Float, const D: usize> {
    controls_p: Vec<Point<T, D>>,
    controls_q: Vec<Point<T, D>>,
    strategy: MLSStrategy,
}

impl<T: Float, const D: usize> PointMLS<T, D> {
    pub fn new<P: Into<Point<T, D>>>(
        controls_p: Vec<P>,
        controls_q: Vec<P>,
    ) -> Result<Self, ConstructionError> {
        if controls_p.len() != controls_q.len() {
            return Err(ConstructionError::MismatchedControlPoints(
                controls_p.len(),
                controls_q.len(),
            ));
        }
        if controls_q.is_empty() {
            return Err(ConstructionError::NoControlPoints);
        }
        Ok(Self {
            controls_p: controls_p.into_iter().map(|p| p.into()).collect(),
            controls_q: controls_q.into_iter().map(|p| p.into()).collect(),
            strategy: MLSStrategy::Affine,
        })
    }

    pub fn controls_p(&self) -> &[Point<T, D>] {
        &self.controls_p
    }

    pub fn controls_q(&self) -> &[Point<T, D>] {
        &self.controls_q
    }

    pub fn strategy(&self) -> &MLSStrategy {
        &self.strategy
    }

    pub fn transform<P: Into<Point<T, D>>>(&self, p: P) -> [T; D] {
        match self.strategy {
            MLSStrategy::Affine => {
                deform_affine(&self.controls_p, &self.controls_q, p.into()).into()
            }
        }
    }

    pub fn transform_r<P: Into<Point<T, D>>>(&self, p: P) -> [T; D] {
        match self.strategy {
            MLSStrategy::Affine => {
                deform_affine(&self.controls_q, &self.controls_p, p.into()).into()
            }
        }
    }
}

// fn sum_p<const D: usize>(a: Point<Precision, D>, b: Point<Precision, D>) -> Point<Precision, D> {
//     a + b.coords
// }

// fn sum_m<const D: usize>(
//     a: SMatrix<Precision, D, D>,
//     b: SMatrix<Precision, D, D>,
// ) -> SMatrix<Precision, D, D> {
//     a + b
// }

/// To look like reference impl's Point::transpose_mul(Mat2)
/// Actually just matrix.transpose * vector
fn transpose_mul<T: Float, const D: usize>(v: Point<T, D>, m: SMatrix<T, D, D>) -> Point<T, D> {
    m.transpose() * v
}

fn times_transpose<T: Float, const D: usize>(
    a: &SVector<T, D>,
    b: &SVector<T, D>,
) -> SMatrix<T, D, D> {
    a * b.transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{assert_eq_sl, fake_points, make_rng, read_cps, ref_deform};

    #[test]
    fn can_construct() {
        let (c_p, c_q) = read_cps::<3>();
        PointMLS::new(c_p, c_q).expect("Could not construct");
    }

    #[test]
    fn vs_reference() {
        let (c_p, c_q) = read_cps::<2>();
        let mut rng = make_rng();
        let orig = fake_points(&c_p, 100, &mut rng);

        let ref_deformed = ref_deform(&c_p, &c_q, &orig);

        let mls = PointMLS::new(c_p, c_q).expect("Could not construct");

        let mls_deformed: Vec<_> = orig.iter().map(|p| mls.transform(*p)).collect();

        assert_eq_sl(&mls_deformed, &ref_deformed)
    }
}
