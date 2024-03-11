/********************************************************************************
 * @brief Implementation of simple neural network in C++
 ********************************************************************************/
#include <vector>
#include <neural_network.hpp>
#include "button.hpp"
#include "led.hpp"

namespace rpi = yrgo::rpi;

using namespace yrgo::machine_learning;

int main(void) {

    
    rpi::Led led1{17};
    rpi::Button button1{22};
    rpi::Button button2{23};
    rpi::Button button3{24};
    rpi::Button button4{25};
    
    
    const std::vector<std::vector<double>> train_input{
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
        {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
        {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
        {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}};

    const std::vector<std::vector<double>> train_output{
        {0}, {1}, {1}, {0},
        {1}, {0}, {0}, {1},
        {1}, {0}, {0}, {1},
        {0}, {1}, {1}, {0}};

    NeuralNetwork network{4, 10, 1, ActFunc::kTanh};
    network.AddTrainingData(train_input, train_output);
    if (network.Train(100000, 0.01)) {
        network.PrintPredictions(train_input, 2);

    }

    while(true)
    {
        std::vector<double>input(4,0);
        input[0] = button1.isPressed() ? 1:0;
        input[1] = button2.isPressed() ? 1:0;
        input[2] = button3.isPressed() ? 1:0;
        input[3] = button4.isPressed() ? 1:0;

        const auto output{network.Predict(input)};

        if (static_cast<int>(output[0] + 0.5) > 0) {
            led1.on();
    }   else {
            led1.off();
    }


    }
    return 0;
}
