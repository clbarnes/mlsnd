use mlsnd::{PointMLS, Float};

fn print_empty<T: Float, const D: usize>() {
    let ps = vec![[T::zero(); D]];
    let ps2 = ps.clone();
    let mls = PointMLS::new(ps, ps2);
    println!("{:?}", mls);
}

fn main() {
    print_empty::<f32, 2>();
    print_empty::<f32, 3>();
    print_empty::<f32, 4>();
    print_empty::<f32, 5>();
    print_empty::<f32, 6>();
    print_empty::<f32, 7>();
    print_empty::<f32, 8>();

    print_empty::<f64, 2>();
    print_empty::<f64, 3>();
    print_empty::<f64, 4>();
    print_empty::<f64, 5>();
    print_empty::<f64, 6>();
    print_empty::<f64, 7>();
    print_empty::<f64, 8>();
}
