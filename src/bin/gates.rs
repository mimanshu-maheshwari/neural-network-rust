use mm_nn::nn::{T, NNArch, NNMatrix};
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
        cost = calc_cost(&mut model, &df_input, &df_output)
    );

    for _ in 0.._iters {
        finite_diff(&mut model, &mut gradient, &df_input, &df_output, eps);
        learn(&mut model, &mut gradient, rate);
    }
    println!(
        "cost: {cost}",
        cost = calc_cost(&mut model, &df_input, &df_output)
    );
    check_output(&mut model, &df_input, &df_output);
}

/// use finite difference method to create gradient value 
/// cost = lim(x -> 0) {f(w + h) - f(w) / h}
fn finite_diff(
    model: &mut NNArch,
    gradient: &mut NNArch,
    df_input: &NNMatrix,
    df_output: &NNMatrix,
    eps: T,
) {
    let mut saved: T;

    let cost = calc_cost(model, df_input, df_output);

    for i in 0..model.w1.row {
        for j in 0..model.w1.col {
            saved = model.w1.get_at(i, j);
            *model.w1.get_mut_at(i, j) += eps;
            *gradient.w1.get_mut_at(i, j) = (calc_cost(model, df_input, df_output) - cost) / eps;
            *model.w1.get_mut_at(i, j) = saved;
        }
    }

    for i in 0..model.b1.row {
        for j in 0..model.b1.col {
            saved = model.b1.get_at(i, j);
            *model.b1.get_mut_at(i, j) += eps;
            *gradient.b1.get_mut_at(i, j) = (calc_cost(model, df_input, df_output) - cost) / eps;
            *model.b1.get_mut_at(i, j) = saved;
        }
    }

    for i in 0..model.w2.row {
        for j in 0..model.w2.col {
            saved = model.w2.get_at(i, j);
            *model.w2.get_mut_at(i, j) += eps;
            *gradient.w2.get_mut_at(i, j) = (calc_cost(model, df_input, df_output) - cost) / eps;
            *model.w2.get_mut_at(i, j) = saved;
        }
    }

    for i in 0..model.b2.row {
        for j in 0..model.b2.col {
            saved = model.b2.get_at(i, j);
            *model.b2.get_mut_at(i, j) += eps;
            *gradient.b2.get_mut_at(i, j) = (calc_cost(model, df_input, df_output) - cost) / eps;
            *model.b2.get_mut_at(i, j) = saved;
        }
    }
}

/// use the gradient to change the values of model.
/// model(w_n) -= gradient(w_n) * rate
/// model(b_n) -= gradient(b_n) * rate
fn learn(model: &mut NNArch, gradient: &NNArch, rate: T) {
    for i in 0..model.w1.row {
        for j in 0..model.w1.col {
            *model.w1.get_mut_at(i, j) -= rate * gradient.w1.get_at(i, j);
        }
    }

    for i in 0..model.b1.row {
        for j in 0..model.b1.col {
            *model.b1.get_mut_at(i, j) -= rate * gradient.b1.get_at(i, j);
        }
    }

    for i in 0..model.w2.row {
        for j in 0..model.w2.col {
            *model.w2.get_mut_at(i, j) -= rate * gradient.w2.get_at(i, j);
        }
    }

    for i in 0..model.b2.row {
        for j in 0..model.b2.col {
            *model.b2.get_mut_at(i, j) -= rate * gradient.b2.get_at(i, j);
        }
    }
}

fn forward(model: &mut NNArch) {
    model.a1 = &model.a0 * &model.w1;
    model.a1 += &model.b1;
    model.a1.sigmoid();
    model.a2 = &model.a1 * &model.w2;
    model.a2 += &model.b2;
    model.a2.sigmoid();
}

fn calc_cost(model: &mut NNArch, df_input: &NNMatrix, df_output: &NNMatrix) -> T {
    let mut cost: T = 0.0;
    for i in 0..df_input.row {
        model.a0.copy_row_from(df_input, i);
        forward(model);
        let result: T = model.a2.get_at(0, 0);
        let o0: T = df_output.get_at(i, 0);
        cost += (o0 - result) * (o0 - result);
    }
    cost / (df_input.row as T)
}

fn check_output(model: &mut NNArch, df_input: &NNMatrix, df_output: &NNMatrix) {
    for i in 0..df_input.row {
        model.a0.copy_row_from(df_input, i);
        forward(model);
        println!(
            "{input:?}: {actual} | {expected}",
            input = df_input.get_row(i),
            actual = model.a2.get_at(0, 0),
            expected = df_output.get_at(i, 0)
        );
    }
}
