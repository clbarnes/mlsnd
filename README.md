# mlsnd: Moving Least Squares in N Dimensions

An implementation of the moving least squares point deformation algorithm ([Schaefer 2006](https://people.engr.tamu.edu/schaefer/research/mls.pdf)).

Heavily inspired by the existing [`moving-least-squares`](https://crates.io/crates/moving-least-squares) crate. Here is how they compare:

| Feature           | `moving-least-squares`    | `mlsnd`                   |
|-------------------|---------------------------|---------------------------|
| Number types      | f32 only                  | Generic over f32, f64     |
| Dimensionality    | 2D only                   | Generic over N dimensions |
| Speed             | ~20% faster               | Slower                    |
| Algorithm support | Affine, rigid, similarity | Affine only               |
| Dependencies      | Fewer                     | More (mainly `nalgebra`)  |
| Results (2D f32)  | Same (near enough)        | Same (near enough)        |
