pub type T = f32;

use rand::Rng;
use std::fmt;
use std::ops;
use std::ops::{Add, AddAssign, Mul, MulAssign};

#[derive(Debug, PartialEq)]
pub struct NNMatrix {
    pub data_frame: Box<[T]>,
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
}

impl NNMatrix {
    /// create a new matrix of row , column from a linier contiguous array
    /// eg. =>
    ///     a ^ b = c
    ///     [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0]
    ///     Here first index value is input a second in input b and third is output create
    ///     this create for rows.
    ///     Here stride is 3 as a complete row is of size 3
    pub fn new(df: Option<&[T]>, rows: usize, cols: usize, stride: usize) -> Self {
        let data_frame: Box<[T]>;
        if let Some(df) = df {
            data_frame = df.to_vec().into_boxed_slice();
        } else {
            data_frame = NNMatrix::alloc(rows, cols);
        }
        NNMatrix {
            data_frame,
            rows,
            cols,
            stride,
        }
    }

    /// create a empty matrix of 0 for rows and columns
    pub fn empty(rows: usize, cols: usize) -> Self {
        let data_frame = NNMatrix::alloc(rows, cols);
        NNMatrix {
            data_frame,
            rows,
            cols,
            stride: cols,
        }
    }

    pub fn get_at(&self, row: usize, col: usize) -> T {
        assert!(row <= self.rows && col <= self.cols);
        self.data_frame[row * self.stride + col]
    }

    pub fn get_mut_at(&mut self, row: usize, col: usize) -> &mut T {
        assert!(row * self.stride + col < self.data_frame.len());
        &mut self.data_frame[row * self.stride + col]
    }

    pub fn set_at(&mut self, row: usize, col: usize, value: T) {
        assert!(row <= self.rows && col <= self.cols);
        if let Some(elem) = self.data_frame.get_mut(row * self.stride + col) {
            *elem = value;
        }
    }

    pub fn rand_range(&mut self, range: ops::Range<T>) {
        let mut rng = rand::thread_rng();
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set_at(i, j, rng.gen_range(range.clone()));
            }
        }
    }

    pub fn rand(&mut self) {
        let mut rng = rand::thread_rng();
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set_at(i, j, rng.gen_range(0.0..1.0));
            }
        }
    }

    // pub fn product(&self, b: &NNMatrix, c: &mut NNMatrix) {
    //     let a: &NNMatrix = self;
    //     let n = a.cols;
    //     let row = a.row;
    //     let cols = b.cols;
    //     assert!(a.cols == b.row && a.row == c.row && b.cols == c.cols);
    //     for i in 0..row {
    //         for j in 0..cols {
    //             for k in 0..n {
    //                 *c.get_mut_at(i, j) += a.get_at(i, k) * b.get_at(k, j);
    //             }
    //         }
    //     }
    // }

    /// allocate contiguous space on heap for array.
    fn alloc(rows: usize, cols: usize) -> Box<[T]> {
        let v: Vec<T> = vec![0 as T; rows * cols];
        v.into_boxed_slice()
    }

    pub fn get_row(&self, row: usize) -> Box<[T]> {
        let row = row * self.stride;
        self.data_frame[row..row + self.cols]
            .to_owned()
            .into_boxed_slice()
    }

    pub fn copy_row_to(&self, to: &mut NNMatrix, row: usize) {
        assert!(to.cols == self.cols);
        for i in 0..self.cols {
            *to.get_mut_at(0, i) = self.get_at(row, i);
        }
    }

    pub fn copy_row_from(&mut self, from: &NNMatrix, row: usize) {
        assert!(from.cols == self.cols);
        for i in 0..self.cols {
            *self.get_mut_at(0, i) = from.get_at(row, i);
        }
    }

    pub fn sigmoid(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.get_mut_at(i, j) = sigmoid(self.get_at(i, j));
            }
        }
    }

    pub fn zeroed(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set_at(i, j, 0.0);
            }
        }
    }

    pub fn identity(rows: usize, cols: usize) -> NNMatrix {
        let mut id_matrix: NNMatrix = NNMatrix::empty(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                if i == j {
                    id_matrix.set_at(i, j, 1.0);
                }
            }
        }
        id_matrix
    }
}

// ====================== arithmetic ops start ==================================== //
// ---------------------- product ------------------------------------ //

/// scaler product
impl Mul<T> for &NNMatrix {
    type Output = NNMatrix;
    fn mul(self, b: T) -> NNMatrix {
        let rows = self.rows;
        let cols = self.cols;
        let mut c = NNMatrix::empty(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                *c.get_mut_at(i, j) = self.get_at(i, j) * b;
            }
        }
        c
    }
}
/// dot product
impl Mul<&NNMatrix> for &NNMatrix {
    type Output = NNMatrix;
    fn mul(self, b: &NNMatrix) -> Self::Output {
        let mut c = NNMatrix::empty(self.rows, b.cols);
        assert!(self.cols == b.rows);
        assert!(self.rows == c.rows);
        assert!(b.cols == c.cols);
        let n = self.cols;
        let rows = self.rows;
        let cols = b.cols;
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..n {
                    *c.get_mut_at(i, j) += self.get_at(i, k) * b.get_at(k, j);
                }
            }
        }
        c
    }
}

/// dot product
impl Mul<NNMatrix> for &NNMatrix {
    type Output = NNMatrix;
    fn mul(self, b: NNMatrix) -> Self::Output {
        let mut c = NNMatrix::empty(self.rows, b.cols);
        assert!(self.cols == b.rows && self.rows == c.rows && b.cols == c.cols);
        let n = self.cols;
        let rows = self.rows;
        let cols = b.cols;
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..n {
                    *c.get_mut_at(i, j) += self.get_at(i, k) * b.get_at(k, j);
                }
            }
        }
        c
    }
}
// ---------------------- product assign ------------------------------------ //

/// scaler product assign
impl MulAssign<T> for NNMatrix {
    fn mul_assign(&mut self, b: T) {
        let rows = self.rows;
        let cols = self.cols;
        for i in 0..rows {
            for j in 0..cols {
                *self.get_mut_at(i, j) *= b;
            }
        }
    }
}

/// dot product assign
impl MulAssign<NNMatrix> for NNMatrix {
    fn mul_assign(&mut self, b: NNMatrix) {
        let mut c = NNMatrix::empty(self.rows, b.cols);
        assert!(self.cols == b.rows && self.rows == c.rows && b.cols == c.cols);
        let n = self.cols;
        let rows = self.rows;
        let cols = b.cols;
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..n {
                    *c.get_mut_at(i, j) += self.get_at(i, k) * b.get_at(k, j);
                }
            }
        }
        for i in 0..rows {
            for j in 0..cols {
                *self.get_mut_at(i, j) = c.get_at(i, j);
            }
        }
        drop(c);
    }
}

/// dot product assign
impl MulAssign<&NNMatrix> for NNMatrix {
    fn mul_assign(&mut self, b: &NNMatrix) {
        let mut c = NNMatrix::empty(self.rows, b.cols);
        assert!(self.cols == b.rows && self.rows == c.rows && b.cols == c.cols);
        let n = self.cols;
        let rows = self.rows;
        let cols = b.cols;
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..n {
                    *c.get_mut_at(i, j) += self.get_at(i, k) * b.get_at(k, j);
                }
            }
        }
        for i in 0..rows {
            for j in 0..cols {
                *self.get_mut_at(i, j) = c.get_at(i, j);
            }
        }
        drop(c);
    }
}

// ---------------------- addition ------------------------------------ //
/// scaler addition
impl Add<T> for &NNMatrix {
    type Output = NNMatrix;
    fn add(self, b: T) -> Self::Output {
        let mut c = NNMatrix::empty(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                *c.get_mut_at(i, j) = self.get_at(i, j) + b;
            }
        }
        c
    }
}

/// matrix addition
impl Add<NNMatrix> for &NNMatrix {
    type Output = NNMatrix;
    fn add(self, b: NNMatrix) -> Self::Output {
        assert!(self.cols == b.cols);
        assert!(self.rows == b.rows);
        let mut c = NNMatrix::empty(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                *c.get_mut_at(i, j) = self.get_at(i, j) + b.get_at(i, j);
            }
        }
        c
    }
}

// ---------------------- addition assign ------------------------------------ //
/// matrix add assign
impl AddAssign<NNMatrix> for NNMatrix {
    fn add_assign(&mut self, rhs: NNMatrix) {
        assert!(self.cols == rhs.cols);
        assert!(self.rows == rhs.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.get_mut_at(i, j) += rhs.get_at(i, j);
            }
        }
    }
}

/// matrix add assign
impl AddAssign<&NNMatrix> for NNMatrix {
    fn add_assign(&mut self, rhs: &NNMatrix) {
        assert!(self.cols == rhs.cols);
        assert!(self.rows == rhs.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.get_mut_at(i, j) += rhs.get_at(i, j);
            }
        }
    }
}

/// scaler matrix add assign
impl AddAssign<T> for NNMatrix {
    fn add_assign(&mut self, rhs: T) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.get_mut_at(i, j) += rhs;
            }
        }
    }
}
// ====================== arithmetic ops end ==================================== //

// ====================== display trait start ==================================== //
impl fmt::Display for NNMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        assert!(self.rows * self.cols <= self.data_frame.len());
        writeln!(f).unwrap();
        for row in 0..self.rows {
            for col in 0..self.cols {
                let prefix = if col == 0 { "  |" } else { "," };
                let postfix = if col + 1 == self.cols { "|" } else { "" };
                write!(
                    f,
                    "{prefix}{num:-9.6}{postfix}",
                    num = self.get_at(row, col)
                )
                .unwrap();
            }
            writeln!(f).unwrap();
        }
        Ok(())
    }
}
// ====================== display trait end ==================================== //

#[derive(Debug)]
pub struct NNArch {
    /// the number of layers in the architecture excluding input
    pub layer_count: usize,

    /// activation layers
    /// the amount of activations will be number of layers + 1 as first activation layer(a0) will be the input.
    pub al: Box<[NNMatrix]>,

    /// weights layers
    /// the amount of weights will be number of layers
    pub wl: Box<[NNMatrix]>,

    /// biases layers
    /// the amount of biases will be number of layers
    pub bl: Box<[NNMatrix]>,
}

impl NNArch {
    /// layer_arch will have first layer as input column size, then multiple hiden layers size
    /// and last layer will be output layer size.
    pub fn create(layer_arch: &[usize]) -> Self {
        assert!(layer_arch.len() >= 2);
        let layer_count = layer_arch.len() - 1;
        let mut al: Vec<NNMatrix> = Vec::new();
        let mut wl: Vec<NNMatrix> = Vec::new();
        let mut bl: Vec<NNMatrix> = Vec::new();

        // create input layer
        let a0: NNMatrix = NNMatrix::empty(1, layer_arch[0]);
        al.push(a0);

        // create rest of the layers
        for i in 1..=layer_count {
            let w: NNMatrix = NNMatrix::empty(al[i - 1].cols, layer_arch[i]);
            let b: NNMatrix = NNMatrix::empty(1, layer_arch[i]);
            let a: NNMatrix = NNMatrix::empty(1, layer_arch[i]);

            // push matrix into the layers.
            al.push(a);
            bl.push(b);
            wl.push(w);
        }

        // convert vectors into boxed slices.
        let al = al.into_boxed_slice();
        let bl = bl.into_boxed_slice();
        let wl = wl.into_boxed_slice();

        // return the neural network architecture.
        NNArch {
            layer_count,
            al,
            bl,
            wl,
        }
    }

    pub fn get_input(&self) -> Box<&NNMatrix> {
        Box::new(&self.al[0])
    }

    pub fn get_input_mut(&mut self) -> Box<&mut NNMatrix> {
        Box::new(&mut self.al[0])
    }

    pub fn get_output(&self) -> Box<&NNMatrix> {
        Box::new(&self.al[self.layer_count])
    }

    pub fn get_output_mut(&mut self) -> Box<&mut NNMatrix> {
        Box::new(&mut self.al[self.layer_count])
    }

    pub fn randomize_range(&mut self, range: ops::Range<T>) {
        for m in self.wl.iter_mut() {
            m.rand_range(range.clone());
        }
        for m in self.bl.iter_mut() {
            m.rand_range(range.clone());
        }
    }

    pub fn randomize(&mut self) {
        for m in self.wl.iter_mut() {
            m.rand();
        }
        for m in self.bl.iter_mut() {
            m.rand();
        }
    }

    pub fn zeroed(&mut self) {
        for i in 0..self.layer_count {
            self.al[i].zeroed();
            self.wl[i].zeroed();
            self.bl[i].zeroed();
        }
        self.al[self.layer_count].zeroed();
    }

    /// use finite difference method to create gradient value
    /// cost = lim(x -> 0) {f(w + h) - f(w) / h}
    pub fn finite_diff(
        &mut self,
        gradient: &mut NNArch,
        df_input: &NNMatrix,
        df_output: &NNMatrix,
        eps: T,
    ) {
        let mut saved: T;

        let cost = self.cost(df_input, df_output);

        for i in 0..self.layer_count {
            for row in 0..self.wl[i].rows {
                for col in 0..self.wl[i].cols {
                    saved = self.wl[i].get_at(row, col);
                    *self.wl[i].get_mut_at(row, col) += eps;
                    *gradient.wl[i].get_mut_at(row, col) =
                        (self.cost(df_input, df_output) - cost) / eps;
                    *self.wl[i].get_mut_at(row, col) = saved;
                }
            }
        }

        for i in 0..self.layer_count {
            for row in 0..self.bl[i].rows {
                for col in 0..self.bl[i].cols {
                    saved = self.bl[i].get_at(row, col);
                    *self.bl[i].get_mut_at(row, col) += eps;
                    *gradient.bl[i].get_mut_at(row, col) =
                        (self.cost(df_input, df_output) - cost) / eps;
                    *self.bl[i].get_mut_at(row, col) = saved;
                }
            }
        }
    }

    /// use the gradient to change the values of model.
    /// model(w_n) -= gradient(w_n) * rate
    /// model(b_n) -= gradient(b_n) * rate
    pub fn learn(&mut self, gradient: &NNArch, rate: T) {
        for i in 0..self.layer_count {
            for row in 0..self.wl[i].rows {
                for col in 0..self.wl[i].cols {
                    *self.wl[i].get_mut_at(row, col) -= rate * gradient.wl[i].get_at(row, col);
                }
            }
        }
        for i in 0..self.layer_count {
            for row in 0..self.bl[i].rows {
                for col in 0..self.bl[i].cols {
                    *self.bl[i].get_mut_at(row, col) -= rate * gradient.bl[i].get_at(row, col);
                }
            }
        }
    }

    pub fn backprop(&mut self, gradient: &mut NNArch, df_input: &NNMatrix, df_output: &NNMatrix) {
        assert!(df_input.rows == df_output.rows);
        assert!(df_output.cols == self.get_output().cols);

        let sample_rows: usize = df_input.rows;

        gradient.zeroed();

        // sr_index -> sample row index
        // sc_index -> sample column index
        // l_index -> layer index

        for sr_index in 0..sample_rows {
            self.get_input_mut().copy_row_from(df_input, sr_index);

            self.forward();

            for i in 0..=gradient.layer_count {
                gradient.al[i].zeroed();
            }

            for sc_index in 0..df_output.cols {
                *gradient.get_output_mut().get_mut_at(0, sc_index) =
                    self.get_output().get_at(0, sc_index) - df_output.get_at(sr_index, sc_index);
            }

            for l_index in (1..=self.layer_count).rev() {
                for w_col in 0..self.al[l_index].cols {
                    let a: T = self.al[l_index].get_at(0, w_col);
                    let da: T = gradient.al[l_index].get_at(0, w_col);
                    *gradient.bl[l_index - 1].get_mut_at(0, w_col) += 2.0 * da * a * (1.0 - a);
                    for w_row in 0..self.al[l_index - 1].cols {
                        let pa: T = self.al[l_index - 1].get_at(0, w_row);
                        let w: T = self.wl[l_index - 1].get_at(w_row, w_col);
                        *gradient.wl[l_index - 1].get_mut_at(w_row, w_col) +=
                            2.0 * da * a * (1.0 - a) * pa;
                        *gradient.al[l_index - 1].get_mut_at(0, w_row) +=
                            2.0 * da * a * (1.0 - a) * w;
                    }
                }
            }
            for l_index in 0..gradient.layer_count {
                for row in 0..gradient.wl[l_index].rows {
                    for col in 0..gradient.wl[l_index].cols {
                        *gradient.wl[l_index].get_mut_at(row, col) /= sample_rows as T;
                    }
                }
            }
            for l_index in 0..gradient.layer_count {
                for row in 0..gradient.bl[l_index].rows {
                    for col in 0..gradient.bl[l_index].cols {
                        *gradient.bl[l_index].get_mut_at(row, col) /= sample_rows as T;
                    }
                }
            }
        }
    }

    pub fn forward(&mut self) {
        for i in 0..self.layer_count {
            self.al[i + 1] = &self.al[i] * &self.wl[i];
            self.al[i + 1] += &self.bl[i];
            self.al[i + 1].sigmoid();
        }
    }

    pub fn cost(&mut self, df_input: &NNMatrix, df_output: &NNMatrix) -> T {
        let mut cost: T = 0.0;
        for i in 0..df_input.rows {
            self.get_input_mut().copy_row_from(df_input, i);

            self.forward();

            let result: Box<[T]> = self.get_output().get_row(0);
            let output: Box<[T]> = df_output.get_row(i);

            for col in 0..df_output.cols {
                let diff = output[col] - result[col];
                cost += diff * diff;
            }
        }
        cost / (df_input.rows as T)
    }

    pub fn check_output(&mut self, df_input: &NNMatrix, df_output: &NNMatrix) {
        for i in 0..df_input.rows {
            self.get_input_mut().copy_row_from(df_input, i);
            self.forward();
            println!(
                "{input:?}: {actual:?} | {expected:?}",
                input = df_input.get_row(i),
                actual = self.get_output().get_row(0),
                expected = df_output.get_row(i)
            );
        }
    }
}
impl fmt::Display for NNArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        writeln!(f).unwrap();
        for i in 0..self.layer_count {
            write!(f, "wl{i}:").unwrap();
            writeln!(f, "{layer}", layer = self.wl[i]).unwrap();
            write!(f, "bl{i}:").unwrap();
            writeln!(f, "{layer}", layer = self.bl[i]).unwrap();
        }
        Ok(())
    }
}

pub fn sigmoid(num: T) -> T {
    (1 as T) / ((1 as T) + (-num).exp())
}
