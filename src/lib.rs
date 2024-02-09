
pub mod nn {

    use rand::Rng;

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

        pub fn set_at(&mut self, row: usize, col:usize, value: f32){
            assert!(row <= self.row && col <= self.col);
            if let Some(elem) = self.data_frame.get_mut(row * self.stride + col) {
                *elem = value;
            }
        }
        
        pub fn rand(&mut self, min:f32, max: f32){
            let mut rng = rand::thread_rng();
            for i in 0..self.row {
                for j in 0..self.col{
                    self.set_at(i, j, rng.gen_range(min..max));
                }
            }
        }
    }

    pub mod math {
        pub fn sigmoid(num: f32) -> f32 {
            let out = 1_f32 / (1_f32 + (-num).exp());
            out
        }
    }
}

