use mm_nn::nn::NNArch;
fn main() {
    let mut a: NNArch = NNArch::create(&vec![28 * 28, 16, 16, 10][..]);
    a.randomize_range(0.0..20.0);
    print!("{a}");
}
