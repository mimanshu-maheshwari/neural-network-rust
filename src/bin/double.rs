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
    // learning rate
    let rate: f32 = 1e-1;

    for i in 0..50 {
        let dw = dcost(&w, &data_frame);
        w -= rate * dw;
        let cost = calc_cost(&w, &data_frame);
        println!("{i:-2}: cost={cost:-9.6}, w={w:-9.6}");
        if cost == 0.0 {
            break;
        }
    }
    println!("--------------------------");

    for data in data_frame {
        let expected_output = data[1];
        let actual_output = data[0] * w;
        println!("expected: {expected_output} -> actual: {actual_output}");
    }

    Ok(())
}

fn dcost(w: &f32, data_frame: &[[f32; 2]]) -> f32 {
    let mut result: f32 = 0f32;
    for data in data_frame {
        // a = nw + b;
        let x = data[0];
        let y = data[1];
        result += 2.0 * x * (x * w - y);
    }

    // get calc_cost
    result / data_frame.len() as f32
}
fn calc_cost(w: &f32, data_frame: &[[f32; 2]]) -> f32 {
    let mut result: f32 = 0f32;
    for data in data_frame {
        let x = data[0];
        let y = x * w;
        let d = y - data[1];
        result = d * d;
    }
    result / data_frame.len() as f32
}
