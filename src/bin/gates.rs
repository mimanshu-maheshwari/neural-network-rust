use nn::{NNArch, NNMatrix, T};
use std::env;

fn main() {
    let mut args = env::args();
    let program_name: String = args.next().unwrap_or(String::from("no name found"));
    println!("Running {program_name}");

    // number of iterations
    let iters: usize = match args.next() {
        Some(value) => value.parse::<usize>().unwrap_or(1000),
        None => 1000,
    };

    let _and_data_frame: Vec<T> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let _or_data_frame: Vec<T> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let _xor_data_frame: Vec<T> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];

    // training data
    let td = _xor_data_frame;
    // learning rate
    let rate: T = 1e-1;
    // ε epsilon (limit that tends to 0)
    let _eps: T = 1e-1;

    let df_input: NNMatrix = NNMatrix::new(Some(&td[..]), 4, 2, 3);
    let df_output: NNMatrix = NNMatrix::new(Some(&td[2..]), 4, 1, 3);

    let layer_arch: Vec<usize> = vec![2, 2, 1];

    let mut model = NNArch::create(&layer_arch[..]);
    model.randomize();
    let mut gradient = NNArch::create(&layer_arch[..]);
    gradient.zeroed();

    // println!("input: {df_input}output: {df_output}");
    // println!("model: {model}");
    // println!( "initial cost: {cost}", cost = model.cost(&df_input, &df_output));

    for _ in 0..iters {
        // model.finite_diff(&mut gradient, &df_input, &df_output, _eps);
        model.backprop(&mut gradient, &df_input, &df_output);
        model.learn(&gradient, rate);
        println!("cost: {cost}", cost = model.cost(&df_input, &df_output));
    }

    // println!( "updated cost: {cost}", cost = model.cost(&df_input, &df_output));

    // println!("model: {model}\ngradient: {gradient}");
    model.check_output(&df_input, &df_output);
}
