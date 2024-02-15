use mm_nn::nn::{NNArch, NNMatrix, T};
use std::env;

fn main() {
    let program_name: String = env::args().next().unwrap_or(String::from("no name found"));
    println!("Running {program_name}");

    let _and_data_frame: Vec<T> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let _or_data_frame: Vec<T> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let _xor_data_frame: Vec<T> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];

    // training data
    let td = _xor_data_frame;
    // learning rate
    let rate = 1e-1;
    // ε epsilon (limit that tends to 0)
    let eps = 1e-1;
    // number of iterations
    let _iters = 10000;

    let df_input: NNMatrix = NNMatrix::new(Some(&td[..]), 4, 2, 3);
    let df_output: NNMatrix = NNMatrix::new(Some(&td[2..]), 4, 1, 3);

    let mut model = NNArch::new();
    let mut gradient = NNArch::new();

    println!("input: {df_input}\n\noutput: {df_output}\n");
    println!("model: {model:?}\n\ngradient: {gradient:?}\n");
    println!(
        "cost: {cost}",
        cost = model.calc_cost(&df_input, &df_output)
    );

    for _ in 0.._iters {
        model.finite_diff(&mut gradient, &df_input, &df_output, eps);
        model.learn(&mut gradient, rate);
    }
    println!(
        "cost: {cost}",
        cost = model.calc_cost(&df_input, &df_output)
    );
    check_output(&mut model, &df_input, &df_output);
}

fn check_output(model: &mut NNArch, df_input: &NNMatrix, df_output: &NNMatrix) {
    for i in 0..df_input.row {
        model.a0.copy_row_from(df_input, i);
        model.forward();
        println!(
            "{input:?}: {actual} | {expected}",
            input = df_input.get_row(i),
            actual = model.a2.get_at(0, 0),
            expected = df_output.get_at(i, 0)
        );
    }
}
