#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;

double activate(double S)
{
    return 1 / (1 + exp(-S)); // функция активации - сигмоид
}

vector<vector<double>> readCSV(const string& filename)
{
    vector<vector<double>> data; // двумерный вектор для хранения данных

    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Ошибка при открытии файла!" << endl;
        return data;
    }

    string line;

    // чтение файла построчно
    while (getline(file, line))
    {
        vector<double> row;
        istringstream iss(line);
        string cell;

        // разделение строки на ячейки по разделителю ';'
        while (getline(iss, cell, ';'))
        {
            row.push_back(stod(cell)); // Преобразование строки в число типа double и добавление в вектор 
        }

        data.push_back(row); // Добавление строки в двумерный вектор
    }

    return data;
}

int main()
{
    string new_1 = "weights_1.csv";
    string new_2 = "weights_2.csv";

    vector <vector<double>> weights_1(785, vector<double>(81)); // веса для in -> hidden
    vector<vector<double>> weights_2(81, vector<double>(10)); // веса для hidden -> out

    weights_1 = readCSV (new_1); // веса для in -> hidden
    weights_2 = readCSV(new_2); // веса для hidden -> out
    int choice = 0;


    while (true)
    {
        string filePath;
        cout << "Input path to your picture: ";
        cin >> filePath;

        vector<double> in(785, 0.0); // вход
        vector<double> hidden(81, 0.0); // скрытый слой
        vector <double> out(10, 0.0); // выход

        // считываем изображение с помощью OpenCV
        Mat image = imread(filePath, IMREAD_GRAYSCALE);
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

            hidden[i] = activate(S_hidden[i]); // расчет выхода нейрона слоя "hidden"
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
            out[i] = activate(S_out[i]); // расчет выхода нейрона слоя "out"
           cout << i << ") " << out[i] << endl;
        }
        cout << endl;
        
        auto max_it = max_element(out.begin(), out.end()); // находим итератор на максимальный элемент вектора
        int max_index = distance(out.begin(), max_it); // получаем индекс максимального элемента

        cout << "Number on your picture: " << max_index << endl << endl;

        cout << "Do you want to continue? (1 - yes; 0 - no) ";
        cin >> choice;

        if (choice == 0)
        {
            break;
        }
    }
    return 0;
}

