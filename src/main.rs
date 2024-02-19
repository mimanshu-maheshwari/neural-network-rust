use mm_nn::nn::NNArch;
fn main() {
    let a: NNArch = NNArch::create(&vec![28 * 28, 16, 16, 10][..]);
    print!("{a}");
}
