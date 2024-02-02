use rand::Rng;
use std::result;

type Result<T> = result::Result<T, ()>;

fn main() -> Result<()> {
    let mut rng = rand::thread_rng();

    let data_frame: [[f32; 2]; 4] = [
        [1_f32, 2_f32],
        [2_f32, 4_f32],
        [3_f32, 6_f32],
        [4_f32, 8_f32],
    ];

    // initialize weights
    let mut w: f32 = rng.gen::<f32>();
    let mut b: f32 = rng.gen::<f32>();
    // small nudge to weights
    let eps: f32 = 1e-3;
    // learning rate
    let rate: f32 = 1e-3;

    for i in 0..500 {
        // calculate finite difference
        // finite difference =  lim h->0 (cost(a + h) - cost(a))/ h
        let cost = calc_cost(&w, &b, &data_frame);
        println!("{i}: cost={cost}"); //, w={w}, b={b}");
        let finite_diff_w: f32 = (calc_cost(&(w + eps), &b, &data_frame) - cost)/ eps;
        let finite_diff_b: f32 = (calc_cost(&w, &(b + eps), &data_frame) - cost)/ eps;
        w -= rate * finite_diff_w;
        b -= rate * finite_diff_b;
    }
    println!("--------------------------");

    for data in data_frame {
        let expected_output = data[1]; 
        let actual_output = data[0] * w + b; 
        println!("expected: {expected_output} -> actual: {actual_output}");
    }

    Ok(())
}

fn calc_cost(w: &f32, b: &f32, data_frame: &[[f32; 2]]) -> f32 {
    let mut result: f32 = 0f32;
    for data in data_frame {
        // a = nw + b;
        let n = data[0];
        let o = data[1];
        let a = n * w + b;
        result += (a - o).powi(2);
    }

    // get calc_cost
    result /= data_frame.len() as f32;
    result
}
