pub mod nn {

    use rand::Rng;
    use std::ops::Mul;

    #[derive(Default, Debug)]
    pub struct NNMatrix<'a> {
        pub data_frame: &'a mut [f32],
        pub row: usize,
        pub col: usize,
        pub stride: usize,
    }

    impl<'a> NNMatrix<'a> {
        pub fn new(data_frame: &'a mut [f32], row: usize, col: usize, stride: usize) -> Self {
            NNMatrix {
                data_frame,
                row,
                col,
                stride,
            }
        }

        pub fn get_at(&self, row: usize, col: usize) -> f32 {
            assert!(row <= self.row && col <= self.col);
            self.data_frame[row * self.stride + col]
        }

        pub fn get_mut_at(&mut self, row: usize, col: usize) -> &mut f32 {
            assert!(row <= self.row && col <= self.col);
            &mut self.data_frame[row * self.stride + col]
        }

        pub fn set_at(&mut self, row: usize, col: usize, value: f32) {
            assert!(row <= self.row && col <= self.col);
            if let Some(elem) = self.data_frame.get_mut(row * self.stride + col) {
                *elem = value;
            }
        }

        pub fn rand(&mut self, min: f32, max: f32) {
            let mut rng = rand::thread_rng();
            for i in 0..self.row {
                for j in 0..self.col {
                    self.set_at(i, j, rng.gen_range(min..max));
                }
            }
        }

        pub fn dot(&self, b: &'a NNMatrix, c: &'a mut NNMatrix) {
            let a: &NNMatrix = self;
            assert!(a.row == b.row && a.col == b.col && a.row == c.row && c.col == 1);
            for i in 0..a.row {
                for j in 0..a.col {
                    *c.get_mut_at(i, 0) += a.get_at(i, j) * b.get_at(i, j);
                }
            }
        }

        pub fn cross(&self, b: &NNMatrix, c: &mut NNMatrix) {
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
    }

    impl<'a> Mul<f32> for NNMatrix<'a> {
        type Output = Self;
        fn mul(self, b: f32) -> NNMatrix<'a> {
            let a = self;
            let mut c = NNMatrix::default();
            let row = a.row;
            let col = a.col;
            for i in 0..row {
                for j in 0..col {
                    *c.get_mut_at(i, j) += a.get_at(i, j) * b;
                }
            }
            c
        }
    }

    impl<'a> Mul<NNMatrix<'a>> for NNMatrix<'a> {
        type Output = Self;
        fn mul(self, b: NNMatrix<'a>) -> NNMatrix<'a> {
            let a = self;
            let mut c = NNMatrix::default();
            assert!(a.col == b.row && a.row == c.row && b.col == c.col);
            let n = a.col;
            let row = a.row;
            let col = b.col;
            for i in 0..row {
                for j in 0..col {
                    for k in 0..n {
                        *c.get_mut_at(i, j) += a.get_at(i, k) * b.get_at(k, j);
                    }
                }
            }
            c
        }
    }

    pub mod math {
        pub fn sigmoid(num: f32) -> f32 {
            let out = 1_f32 / (1_f32 + (-num).exp());
            out
        }
    }
}
