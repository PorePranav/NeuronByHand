use rand::Rng;
use std::f64;

#[derive(Debug)]
pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_ih: Vec<Vec<f64>>,
    weights_ho: Vec<Vec<f64>>,
    bias_h: Vec<f64>,
    bias_o: Vec<f64>,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        
        let weights_ih = (0..hidden_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
            
        let weights_ho = (0..output_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
            
        let bias_h = (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias_o = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_ih,
            weights_ho,
            bias_h,
            bias_o,
            learning_rate,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-x))
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }

    pub fn feedforward(&self, input: &[f64]) -> Vec<f64> {
        let hidden: Vec<f64> = (0..self.hidden_size)
            .map(|i| {
                let sum: f64 = (0..self.input_size)
                    .map(|j| self.weights_ih[i][j] * input[j])
                    .sum();
                NeuralNetwork::sigmoid(sum + self.bias_h[i])
            })
            .collect();

        let output: Vec<f64> = (0..self.output_size)
            .map(|i| {
                let sum: f64 = (0..self.hidden_size)
                    .map(|j| self.weights_ho[i][j] * hidden[j])
                    .sum();
                NeuralNetwork::sigmoid(sum + self.bias_o[i])
            })
            .collect();

        output
    }

    pub fn train(&mut self, input: &[f64], target: &[f64]) {
        let hidden: Vec<f64> = (0..self.hidden_size)
            .map(|i| {
                let sum: f64 = (0..self.input_size)
                    .map(|j| self.weights_ih[i][j] * input[j])
                    .sum();
                NeuralNetwork::sigmoid(sum + self.bias_h[i])
            })
            .collect();

        let outputs: Vec<f64> = (0..self.output_size)
            .map(|i| {
                let sum: f64 = (0..self.hidden_size)
                    .map(|j| self.weights_ho[i][j] * hidden[j])
                    .sum();
                NeuralNetwork::sigmoid(sum + self.bias_o[i])
            })
            .collect();

        let output_errors: Vec<f64> = (0..self.output_size)
            .map(|i| target[i] - outputs[i])
            .collect();

        let output_gradients: Vec<f64> = (0..self.output_size)
            .map(|i| output_errors[i] * NeuralNetwork::sigmoid_derivative(outputs[i]))
            .collect();

        let hidden_errors: Vec<f64> = (0..self.hidden_size)
            .map(|i| {
                (0..self.output_size)
                    .map(|j| self.weights_ho[j][i] * output_errors[j])
                    .sum()
            })
            .collect();

        let hidden_gradients: Vec<f64> = (0..self.hidden_size)
            .map(|i| hidden_errors[i] * NeuralNetwork::sigmoid_derivative(hidden[i]))
            .collect();

        for i in 0..self.output_size {
            for j in 0..self.hidden_size {
                self.weights_ho[i][j] += output_gradients[i] * hidden[j] * self.learning_rate;
            }
        }

        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                self.weights_ih[i][j] += hidden_gradients[i] * input[j] * self.learning_rate;
            }
        }

        for i in 0..self.output_size {
            self.bias_o[i] += output_gradients[i] * self.learning_rate;
        }

        for i in 0..self.hidden_size {
            self.bias_h[i] += hidden_gradients[i] * self.learning_rate;
        }
    }
}