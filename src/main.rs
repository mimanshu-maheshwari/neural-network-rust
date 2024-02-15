use mm_nn::nn::NNMatrix;
fn main() {
    let a = NNMatrix::empty(3, 2);
    let b = NNMatrix::empty(2, 3);
    print!("{a}\n{b}");
}
