use nn::NNArch;
fn main() {
    let mut a: NNArch = NNArch::create(&vec![3, 3, 2][..]);
    a.randomize_range(0.0..20.0);
    println!("{a}");
    println!(
        "{input}\n{output}",
        input = a.get_input(),
        output = a.get_output()
    );
}
