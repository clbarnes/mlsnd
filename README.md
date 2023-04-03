# mlsnd: Moving Least Squares in N Dimensions

An implementation of the moving least squares point deformation algorithm ([Schaefer 2006](https://people.engr.tamu.edu/schaefer/research/mls.pdf)).

Heavily inspired by the existing [`moving-least-squares`](https://crates.io/crates/moving-least-squares) crate, but generic over 32- and 64-bit floats, and over different dimensionalities.
If 2D f32 covers your use case, the existing crate has fewer dependencies and is about 20% faster in that case.
