use mm_nn::nn::{self, NNMatrix };
use mm_nn::T;
use rand::Rng;
use std::env;


fn main() {
    let program_name: String = env::args().next().unwrap_or(String::from("no name found"));
    println!("Running {program_name}");
    let mut rng = rand::thread_rng();

    let _and_data_frame: Vec<T> = vec![
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    ];

    let _or_data_frame: Vec<T> = vec![
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let _xor_data_frame: Vec<T> = vec![
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0,
    ];

    let td = _or_data_frame;
    // learning rate
    let rate = 1e-1;
    // ε epsilon (limit that tends to 0)
    let eps = 1e-2;
    // number of iterations
    let iters = 10000;

    let df_input: NNMatrix = NNMatrix::new(Some(&td[..]), 4, 2, 3);
    let df_output: NNMatrix = NNMatrix::new(Some(&td[2..]), 4, 1, 3);

    println!("input: {df_input}\noutput: {df_output}");

    let mut w0: T = rng.gen_range(0.0..=1.0);
    let mut w1: T = rng.gen_range(0.0..=1.0);
    let mut b0: T = rng.gen_range(0.0..=1.0);
    for _i in 0..iters {
        let cost = calc_cost(&df_input, &df_output, &w0, &w1, &b0);
        let finite_diff_w0 = (calc_cost(&df_input, &df_output, &(w0 + eps), &w1, &b0) - cost) / eps;
        let finite_diff_w1 = (calc_cost(&df_input, &df_output, &w0, &(w1 + eps), &b0) - cost) / eps;
        let finite_diff_b0 = (calc_cost(&df_input, &df_output, &w0, &w1, &(b0 + eps)) - cost) / eps;
        w0 -= rate * finite_diff_w0;
        w1 -= rate * finite_diff_w1;
        b0 -= rate * finite_diff_b0;
        //println!("{i}: cost = {cost}");
    }
    check_output(&df_input, &df_output, &w0, &w1, &b0);
}

fn calc_cost(df_input: &NNMatrix, df_output: &NNMatrix, w0: &T, w1: &T, b0: &T) -> T{
    let mut cost: T = 0.0;
    for i in 0..df_input.row {
        let a0: T = df_input.get_at(i, 0);
        let a1: T = df_input.get_at(i, 1);
        let o0: T = df_output.get_at(i, 0);
        let result = nn::math::sigmoid(a0 * w0 + a1 * w1 + b0);
        cost += (o0 - result) * (o0 - result);
    }
    cost / (df_input.row as T)
}

fn check_output(df_input: &NNMatrix, _df_output: &NNMatrix, w0: &T, w1: &T, b0: &T) {
    for i in 0..df_input.row {
        let a0: T = df_input.get_at(i, 0);
        let a1: T = df_input.get_at(i, 1);
        let result = nn::math::sigmoid(a0 * w0 + a1 * w1 + b0);
        println!("{a0} op {a1} = {result}");
    }
}
