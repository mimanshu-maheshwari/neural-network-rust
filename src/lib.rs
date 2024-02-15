pub mod nn {

    pub type T = f32;

    use rand::Rng;
    use std::fmt;
    use std::ops::{Add, AddAssign, Mul, MulAssign};

    #[derive(Debug, PartialEq, Clone)]
    pub struct NNMatrix {
        pub data_frame: Box<[T]>,
        pub row: usize,
        pub col: usize,
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
        pub fn new(df: Option<&[T]>, row: usize, col: usize, stride: usize) -> Self {
            let data_frame: Box<[T]>;
            if let Some(df) = df {
                data_frame = df
                    .iter()
                    .map(|k| k.clone())
                    .collect::<Vec<T>>()
                    .into_boxed_slice();
            } else {
                data_frame = NNMatrix::alloc(row, col);
            }
            NNMatrix {
                data_frame,
                row,
                col,
                stride,
            }
        }

        /// create a empty matrix of 0 for rows and columns
        pub fn empty(row: usize, col: usize) -> Self {
            let data_frame = NNMatrix::alloc(row, col);
            NNMatrix {
                data_frame,
                row,
                col,
                stride: col,
            }
        }

        pub fn get_at(&self, row: usize, col: usize) -> T {
            assert!(row <= self.row && col <= self.col);
            self.data_frame[row * self.stride + col]
        }

        pub fn get_mut_at(&mut self, row: usize, col: usize) -> &mut T {
            assert!(row * self.stride + col < self.data_frame.len());
            &mut self.data_frame[row * self.stride + col]
        }

        pub fn set_at(&mut self, row: usize, col: usize, value: T) {
            assert!(row <= self.row && col <= self.col);
            if let Some(elem) = self.data_frame.get_mut(row * self.stride + col) {
                *elem = value;
            }
        }

        pub fn rand_range(&mut self, min: T, max: T) {
            let mut rng = rand::thread_rng();
            for i in 0..self.row {
                for j in 0..self.col {
                    self.set_at(i, j, rng.gen_range(min..max));
                }
            }
        }

        pub fn rand(&mut self) {
            let mut rng = rand::thread_rng();
            for i in 0..self.row {
                for j in 0..self.col {
                    self.set_at(i, j, rng.gen_range(0.0..1.0));
                }
            }
        }

        // pub fn product(&self, b: &NNMatrix, c: &mut NNMatrix) {
        //     let a: &NNMatrix = self;
        //     let n = a.col;
        //     let row = a.row;
        //     let col = b.col;
        //     assert!(a.col == b.row && a.row == c.row && b.col == c.col);
        //     for i in 0..row {
        //         for j in 0..col {
        //             for k in 0..n {
        //                 *c.get_mut_at(i, j) += a.get_at(i, k) * b.get_at(k, j);
        //             }
        //         }
        //     }
        // }

        /// allocate contiguous space on heap for array.
        fn alloc(row: usize, col: usize) -> Box<[T]> {
            let v: Vec<T> = vec![0 as T; row * col];
            v.into_boxed_slice()
        }

        pub fn get_row(&self, row: usize) -> Box<[T]> {
            let row = row * self.stride;
            self.data_frame[row..row + self.col]
                .to_owned()
                .into_boxed_slice()
        }

        pub fn copy_row_to(&self, to: &mut NNMatrix, row: usize) {
            assert!(to.col == self.col);
            for i in 0..self.col {
                *to.get_mut_at(row, i) = self.get_at(row, i);
            }
        }

        pub fn copy_row_from(&mut self, from: &NNMatrix, row: usize) {
            assert!(from.col == self.col);
            for i in 0..self.col {
                *self.get_mut_at(0, i) = from.get_at(row, i);
            }
        }

        pub fn sigmoid(&mut self) {
            for i in 0..self.row {
                for j in 0..self.col {
                    *self.get_mut_at(i, j) = sigmoid(self.get_at(i, j));
                }
            }
        }
    }

    // ====================== arithmetic ops start ==================================== //
    // ---------------------- product ------------------------------------ //

    /// scaler product
    impl Mul<T> for &NNMatrix {
        type Output = NNMatrix;
        fn mul(self, b: T) -> NNMatrix {
            let row = self.row;
            let col = self.col;
            let mut c = NNMatrix::empty(row, col);
            for i in 0..row {
                for j in 0..col {
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
            let mut c = NNMatrix::empty(self.row, b.col);
            assert!(self.col == b.row);
            assert!(self.row == c.row);
            assert!(b.col == c.col);
            let n = self.col;
            let row = self.row;
            let col = b.col;
            for i in 0..row {
                for j in 0..col {
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
            let mut c = NNMatrix::empty(self.row, b.col);
            assert!(self.col == b.row && self.row == c.row && b.col == c.col);
            let n = self.col;
            let row = self.row;
            let col = b.col;
            for i in 0..row {
                for j in 0..col {
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
            let row = self.row;
            let col = self.col;
            for i in 0..row {
                for j in 0..col {
                    *self.get_mut_at(i, j) *= b;
                }
            }
        }
    }

    /// dot product assign
    impl MulAssign<NNMatrix> for NNMatrix {
        fn mul_assign(&mut self, b: NNMatrix) {
            let mut c = NNMatrix::empty(self.row, b.col);
            assert!(self.col == b.row && self.row == c.row && b.col == c.col);
            let n = self.col;
            let row = self.row;
            let col = b.col;
            for i in 0..row {
                for j in 0..col {
                    for k in 0..n {
                        *c.get_mut_at(i, j) += self.get_at(i, k) * b.get_at(k, j);
                    }
                }
            }
            for i in 0..row {
                for j in 0..col {
                    *self.get_mut_at(i, j) = c.get_at(i, j);
                }
            }
            drop(c);
        }
    }

    /// dot product assign
    impl MulAssign<&NNMatrix> for NNMatrix {
        fn mul_assign(&mut self, b: &NNMatrix) {
            let mut c = NNMatrix::empty(self.row, b.col);
            assert!(self.col == b.row && self.row == c.row && b.col == c.col);
            let n = self.col;
            let row = self.row;
            let col = b.col;
            for i in 0..row {
                for j in 0..col {
                    for k in 0..n {
                        *c.get_mut_at(i, j) += self.get_at(i, k) * b.get_at(k, j);
                    }
                }
            }
            for i in 0..row {
                for j in 0..col {
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
            let mut c = NNMatrix::empty(self.row, self.col);
            for i in 0..self.row {
                for j in 0..self.col {
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
            assert!(self.col == b.col);
            assert!(self.row == b.row);
            let mut c = NNMatrix::empty(self.row, self.col);
            for i in 0..self.row {
                for j in 0..self.col {
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
            assert!(self.col == rhs.col);
            assert!(self.row == rhs.row);
            for i in 0..self.row {
                for j in 0..self.col {
                    *self.get_mut_at(i, j) += rhs.get_at(i, j);
                }
            }
        }
    }

    /// matrix add assign
    impl AddAssign<&NNMatrix> for NNMatrix {
        fn add_assign(&mut self, rhs: &NNMatrix) {
            assert!(self.col == rhs.col);
            assert!(self.row == rhs.row);
            for i in 0..self.row {
                for j in 0..self.col {
                    *self.get_mut_at(i, j) += rhs.get_at(i, j);
                }
            }
        }
    }

    /// scaler matrix add assign
    impl AddAssign<T> for NNMatrix {
        fn add_assign(&mut self, rhs: T) {
            for i in 0..self.row {
                for j in 0..self.col {
                    *self.get_mut_at(i, j) += rhs;
                }
            }
        }
    }
    // ====================== arithmetic ops end ==================================== //

    // ====================== display trait start ==================================== //
    impl fmt::Display for NNMatrix {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
            assert!(self.row * self.col <= self.data_frame.len());
            writeln!(
                f,
                "-- {row} rows, {col} columns --",
                row = self.row,
                col = self.col
            )
            .unwrap();
            for row in 0..self.row {
                for col in 0..self.col {
                    write!(f, " {num}", num = self.get_at(row, col)).unwrap();
                }
                writeln!(f, "").unwrap();
            }
            writeln!(f, "-------------------------------").unwrap();
            Ok(())
        }
    }
    // ====================== display trait end ==================================== //

    #[derive(Debug, Clone)]
    pub struct NNArch {
        // /// activation layers
        // al: Box<[NNMatrix]>,
        // /// weights layers
        // wl: Box<[NNMatrix]>,
        // /// biases layers
        // bl: Box<[NNMatrix]>,

        // input
        pub a0: NNMatrix,

        // layer 1
        pub w1: NNMatrix,
        pub b1: NNMatrix,
        pub a1: NNMatrix,

        // layer 2
        pub w2: NNMatrix,
        pub b2: NNMatrix,
        pub a2: NNMatrix,
    }

    impl NNArch {
        pub fn new() -> Self {
            let a0: NNMatrix = NNMatrix::empty(1, 2);

            let mut w1: NNMatrix = NNMatrix::empty(2, 2);
            let mut b1: NNMatrix = NNMatrix::empty(1, 2);
            let a1: NNMatrix = NNMatrix::empty(1, 2);

            let mut w2: NNMatrix = NNMatrix::empty(2, 1);
            let mut b2: NNMatrix = NNMatrix::empty(1, 1);
            let a2: NNMatrix = NNMatrix::empty(1, 1);

            w1.rand();
            b1.rand();

            w2.rand();
            b2.rand();

            NNArch {
                a0,
                w1,
                b1,
                a1,
                w2,
                b2,
                a2,
            }
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

            let cost = self.calc_cost(df_input, df_output);

            for i in 0..self.w1.row {
                for j in 0..self.w1.col {
                    saved = self.w1.get_at(i, j);
                    *self.w1.get_mut_at(i, j) += eps;
                    *gradient.w1.get_mut_at(i, j) =
                        (self.calc_cost(df_input, df_output) - cost) / eps;
                    *self.w1.get_mut_at(i, j) = saved;
                }
            }

            for i in 0..self.b1.row {
                for j in 0..self.b1.col {
                    saved = self.b1.get_at(i, j);
                    *self.b1.get_mut_at(i, j) += eps;
                    *gradient.b1.get_mut_at(i, j) =
                        (self.calc_cost(df_input, df_output) - cost) / eps;
                    *self.b1.get_mut_at(i, j) = saved;
                }
            }

            for i in 0..self.w2.row {
                for j in 0..self.w2.col {
                    saved = self.w2.get_at(i, j);
                    *self.w2.get_mut_at(i, j) += eps;
                    *gradient.w2.get_mut_at(i, j) =
                        (self.calc_cost(df_input, df_output) - cost) / eps;
                    *self.w2.get_mut_at(i, j) = saved;
                }
            }

            for i in 0..self.b2.row {
                for j in 0..self.b2.col {
                    saved = self.b2.get_at(i, j);
                    *self.b2.get_mut_at(i, j) += eps;
                    *gradient.b2.get_mut_at(i, j) =
                        (self.calc_cost(df_input, df_output) - cost) / eps;
                    *self.b2.get_mut_at(i, j) = saved;
                }
            }
        }

        /// use the gradient to change the values of model.
        /// model(w_n) -= gradient(w_n) * rate
        /// model(b_n) -= gradient(b_n) * rate
        pub fn learn(&mut self, gradient: &NNArch, rate: T) {
            for i in 0..self.w1.row {
                for j in 0..self.w1.col {
                    *self.w1.get_mut_at(i, j) -= rate * gradient.w1.get_at(i, j);
                }
            }

            for i in 0..self.b1.row {
                for j in 0..self.b1.col {
                    *self.b1.get_mut_at(i, j) -= rate * gradient.b1.get_at(i, j);
                }
            }

            for i in 0..self.w2.row {
                for j in 0..self.w2.col {
                    *self.w2.get_mut_at(i, j) -= rate * gradient.w2.get_at(i, j);
                }
            }

            for i in 0..self.b2.row {
                for j in 0..self.b2.col {
                    *self.b2.get_mut_at(i, j) -= rate * gradient.b2.get_at(i, j);
                }
            }
        }

        pub fn forward(&mut self) {
            self.a1 = &self.a0 * &self.w1;
            self.a1 += &self.b1;
            self.a1.sigmoid();
            self.a2 = &self.a1 * &self.w2;
            self.a2 += &self.b2;
            self.a2.sigmoid();
        }

        pub fn calc_cost(&mut self, df_input: &NNMatrix, df_output: &NNMatrix) -> T {
            let mut cost: T = 0.0;
            for i in 0..df_input.row {
                self.a0.copy_row_from(df_input, i);
                self.forward();
                let result: T = self.a2.get_at(0, 0);
                let o0: T = df_output.get_at(i, 0);
                cost += (o0 - result) * (o0 - result);
            }
            cost / (df_input.row as T)
        }
    }

    pub fn sigmoid(num: T) -> T {
        let out = (1 as T) / ((1 as T) + (-num).exp());
        out
    }
}
