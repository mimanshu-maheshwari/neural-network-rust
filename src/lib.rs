pub mod nn {

    #[derive(Default, Debug)]
    pub struct NNMatrix<'a> {
        pub data_frame: &'a [f32],
        pub row: usize,
        pub col: usize,
        pub stride: usize,
    }

    impl<'a> NNMatrix<'a> {
        pub fn new(data_frame: &'a [f32], row: usize, col: usize) -> Self {
            NNMatrix {
                data_frame,
                row,
                col,
                stride: col,
            }
        }

        fn at(&self, row: usize, col: usize) -> f32 {
            assert!(row <= self.row && col <= self.col);
            self.data_frame[row * self.col + col]
        }

        pub fn row_input(&self, row: usize, col: usize) -> f32 {
            assert!(row <= self.row && col <= self.col);
            self.at(row, col)
        }

        pub fn row_input_slice(&self, row: usize) -> &[f32] {
            assert!(row <= self.row);
            &self.data_frame[(row * self.col)..(row * self.col + self.stride - 1)]
        }

        pub fn row_output(&self, row: usize) -> f32 {
            assert!(row <= self.row);
            self.at(row, self.stride - 1)
        }
    }

    pub mod math {
        pub fn sigmoid(num: f32) -> f32 {
            let out = 1_f32 / (1_f32 + (-num).exp());
            out
        }

    }
}

#[cfg(test)]
mod test {
    use super::nn::math;

    #[test]
    fn sigmoid_test_0() {
        let actual = math::sigmoid(0_f32); 
        let expected = 0.5f32;
        assert_eq!(expected, actual);
    }
  
    #[test]
    fn sigmoid_test_1() {
        let actual = math::sigmoid(1_f32); 
        let expected = 0.73105857_f32;
        assert_eq!(expected, actual);
    }

    #[test]
    fn sigmoid_test_minus_1() {
        let actual = math::sigmoid(-1_f32); 
        let expected = 0.268941421_f32;
        assert_eq!(expected, actual);
    }
    
   
}
