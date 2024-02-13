pub type T = f32;

pub mod nn {

    use super::T;
    use rand::Rng;
    use std::fmt;
    use std::ops::{Add, Mul};

    #[derive(Debug, PartialEq)]
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
            assert!(row <= self.row && col <= self.col);
            &mut self.data_frame[row * self.stride + col]
        }

        pub fn set_at(&mut self, row: usize, col: usize, value: T) {
            assert!(row <= self.row && col <= self.col);
            if let Some(elem) = self.data_frame.get_mut(row * self.stride + col) {
                *elem = value;
            }
        }

        pub fn rand(&mut self, min: T, max: T) {
            let mut rng = rand::thread_rng();
            for i in 0..self.row {
                for j in 0..self.col {
                    self.set_at(i, j, rng.gen_range(min..max));
                }
            }
        }

        pub fn product(&self, b: &NNMatrix, c: &mut NNMatrix) {
            let a: &NNMatrix = self;
            let n = a.col;
            let row = a.row;
            let col = b.col;
            assert!(a.col == b.row && a.row == c.row && b.col == c.col);
            for i in 0..row {
                for j in 0..col {
                    for k in 0..n {
                        *c.get_mut_at(i, j) += a.get_at(i, k) * b.get_at(k, j);
                    }
                }
            }
        }

        /// allocate contiguous space on heap for array.
        fn alloc(row: usize, col: usize) -> Box<[T]> {
            let v: Vec<T> = vec![0 as T; row * col];
            v.into_boxed_slice()
        }
    }

    /// scaler product
    impl Mul<T> for NNMatrix {
        type Output = Self;
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
    impl Mul<NNMatrix> for NNMatrix {
        type Output = Self;
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

    /// scaler addition
    impl Add<T> for NNMatrix {
        type Output = Self;
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
    impl Add<NNMatrix> for NNMatrix {
        type Output = Self;
        fn add(self, b: NNMatrix) -> Self::Output {
            assert!(self.col == b.col && self.row == b.row);
            let mut c = NNMatrix::empty(self.row, self.col);
            for i in 0..self.row {
                for j in 0..self.col {
                    *c.get_mut_at(i, j) = self.get_at(i, j) + b.get_at(i, j);
                }
            }
            c
        }
    }

    impl fmt::Display for NNMatrix {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
            assert!(self.row * self.col <= self.data_frame.len());
            writeln!(f, "-- {row} rows, {col} columns --", row = self.row, col = self.col).unwrap();
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

    pub mod math {

        use super::T;

        pub fn sigmoid(num: T) -> T {
            let out = (1 as T) / ((1 as T) + (-num).exp());
            out
        }
    }
}
