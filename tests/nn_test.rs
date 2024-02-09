#[cfg(test)]
pub mod nn_tests {
    use mm_nn::nn::math;

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
