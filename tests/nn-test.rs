#[cfg(test)]
mod nn {

    mod math {
        use nn::sigmoid;
        #[test]
        fn sigmoid_test_0() {
            let actual = sigmoid(0_f32);
            let expected = 0.5f32;
            assert_eq!(expected, actual);
        }

        #[test]
        fn sigmoid_test_1() {
            let actual = sigmoid(1_f32);
            let expected = 0.73105857_f32;
            assert_eq!(expected, actual);
        }

        #[test]
        fn sigmoid_test_minus_1() {
            let actual = sigmoid(-1_f32);
            let expected = 0.268941421_f32;
            assert_eq!(expected, actual);
        }
    }
}
