use mm_nn::nn::{self, NNMatrix};
use rand::Rng;
use std::env;

fn main() {
    let program_name: String = env::args().next().unwrap_or(String::from("no name found"));
    println!("Running {program_name}");
    let mut rng = rand::thread_rng();

    let _and_data_frame: [f32; 12] = [
        0f32, 0f32, 0f32, 0f32, 1f32, 0f32, 1f32, 0f32, 0f32, 1f32, 1f32, 1f32,
    ];

    let _or_data_frame: [f32; 12] = [
        0f32, 0f32, 0f32, 0f32, 1f32, 1f32, 1f32, 0f32, 1f32, 1f32, 1f32, 1f32,
    ];
    let _xor_data_frame: [f32; 12] = [
        0f32, 0f32, 0f32, 0f32, 1f32, 1f32, 1f32, 0f32, 1f32, 1f32, 1f32, 0f32,
    ];

    let td = _or_data_frame;
    // learning rate
    let rate = 1e-1;
    // ε epsilon (limit that tends to 0)
    let eps = 1e-2;
    // number of iterations
    let iters = 10000;

    // training data frame containing output and input
    let mut td_input = td.clone();
    let mut td_output = td.clone();

    let df_input: NNMatrix = NNMatrix::new(&mut td_input, 4, 2, 3);
    let df_output: NNMatrix = NNMatrix::new(&mut td_output[2..], 4, 1, 3);

    let mut w0: f32 = rng.gen_range(0f32..=1f32);
    let mut w1: f32 = rng.gen_range(0f32..=1f32);
    let mut b0: f32 = rng.gen_range(0f32..=1f32);
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

fn calc_cost(df_input: &NNMatrix, df_output: &NNMatrix, w0: &f32, w1: &f32, b0: &f32) -> f32 {
    let mut cost: f32 = 0f32;
    for i in 0..df_input.row {
        let a0: f32 = df_input.get_at(i, 0);
        let a1: f32 = df_input.get_at(i, 1);
        let o0: f32 = df_output.get_at(i, 0);
        let result = nn::math::sigmoid(a0 * w0 + a1 * w1 + b0);
        cost += (o0 - result) * (o0 - result);
    }
    cost / (df_input.row as f32)
}

fn check_output(df_input: &NNMatrix, _df_output: &NNMatrix, w0: &f32, w1: &f32, b0: &f32) {
    for i in 0..df_input.row {
        let a0: f32 = df_input.get_at(i, 0);
        let a1: f32 = df_input.get_at(i, 1);
        let result = nn::math::sigmoid(a0 * w0 + a1 * w1 + b0);
        println!("{a0} op {a1} = {result}");
    }
}
