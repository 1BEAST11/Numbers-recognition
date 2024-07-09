#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <random>
#include <string>
#include <fstream>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

double sigmoid(double S)
{
    return 1 / (1 + exp(-S)); // функция активации - сигмоид
}

int main()
{
    vector <vector<double>> weights_1(785, vector<double>(81)); // веса для in -> hidden
    vector <vector<double>> weights_2(81, vector<double>(10)); // веса для hidden -> out

    random_device rd;
    mt19937 generator(rd()); // генератор случайных чисел
    uniform_real_distribution<double> distribution(-0.5, 0.5);

    // генеририруем веса для in -> hidden
    for (int i = 0; i < 785; i++)
    {
        for (int j = 0; j < 81; j++)
        {
            weights_1[i][j] = distribution(generator); // веса связей между нейронами
        }
    }

    // генерируем веса для hidden->out
    for (int i = 0; i < 81; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            weights_2[i][j] = distribution(generator); // веса связей между нейронами
        }
    }

    for (int epoch = 0; epoch < 10; epoch++)
    {
        cout << epoch << ") " << endl;
        string folderPath = "training";

        // цикл для поочередного чтения файлов из папки
        for (const auto& entry : fs::directory_iterator(folderPath))
        {
            vector<double> in(785, 0.0); // вход
            vector<double> hidden(81, 0.0); // скрытый слой

            vector<double> out(10, 0.0); // выход
            vector<double> target(10, 0.0); // целевое значение

            // получаем путь к текущему файлу
            string filePath = entry.path().string();
            cout << "Reading file: " << filePath << endl;

            char ch = filePath[filePath.size() - 5];
            int number = (ch == '0') ? 0 : ch - '0';
            target[number] = 1.0;

            for (int i = 0; i < target.size(); i++)
            {
                cout << target[i] << " ";
            }
            cout << endl;

            // ПРЯМОЕ РАСПРОСТРАНЕНИЕ
            Mat image = imread(filePath, IMREAD_GRAYSCALE); // считываем изображение с помощью OpenCV

            threshold(image, image, 0, 1, THRESH_BINARY);
            int count = 0;

            for (int i = 0; i < image.rows; ++i)
            {
                for (int j = 0; j < image.cols; ++j)
                {
                    double pixel = static_cast<double>(image.at<uchar>(i, j));
                    in[count] = pixel; // выходы нейронов слоя слоя "in"
                    count++;
                }
            }
            in[784] = 1.0; // нейрон смещения слоя "in"

            // расчет значений для скрытого слоя
            vector <double> S_hidden(81, 0.0);
            for (int i = 0; i < hidden.size(); i++)
            {
                for (int j = 0; j < in.size(); j++)
                {
                    S_hidden[i] += in[j] * weights_1[j][i]; // расчет входа нейрона слоя "hidden"
                }

                hidden[i] = sigmoid(S_hidden[i]); // расчет выхода нейрона слоя "hidden"
            }
            hidden[80] = 1.0;

            // расчет значений для выходного слоя
            vector <double> S_out(10, 0.0);
            for (int i = 0; i < out.size(); i++)
            {
                for (int j = 0; j < hidden.size(); j++)
                {
                    S_out[i] += hidden[j] * weights_2[j][i]; // расчет входа нейрона слоя "out"
                }
                out[i] = sigmoid(S_out[i]);
                cout << out[i] << endl;
            }
            cout << endl;

            // ОБРАТНОЕ РАСПРОСТРАНЕНИЕ (BACK PROPAGATION)
            double lambda = 0.1;
            vector <double> dE_dS_1(81, 0.0);

            // изменение весов для hidden -> out
            for (int i = 0; i < out.size(); i++)
            {
                double error_2 = 2 * (target[i] - out[i]) * out[i] * (1 - out[i]);

                for (int j = 0; j < hidden.size(); j++)
                {
                    weights_2[j][i] += lambda * error_2 * hidden[j];
                    dE_dS_1[j] += weights_2[j][i] * error_2;
                }
            }

            // изменение весов для in -> hidden
            for (int i = 0; i < hidden.size(); i++)
            {
                double error_1 = hidden[i] * (1 - hidden[i]) * dE_dS_1[i];
                for (int j = 0; j < in.size(); j++)
                {
                    weights_1[j][i] += lambda * error_1 * in[j];

                }
            }
        }
    }

    ofstream new_1("weights_1.csv");
    ofstream new_2("weights_2.csv");

    //  запись весов в файл
    for (int i = 0; i < 785; i++)
    {
        for (int j = 0; j < 81; j++)
        {
            new_1 << setprecision(32) << weights_1[i][j] << ";";
        }
        new_1 << "\n";
    }

    for (int i = 0; i < 81; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            new_2 << setprecision(32) << weights_2[i][j] << ";";
        }
        new_2 << "\n";
    }
    
    cout << "Training complete!" << endl;

    new_1.close();
    new_2.close();

    return 0;
}

