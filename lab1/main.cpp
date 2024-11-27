#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;


int verticalProjection(Mat& image) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    
    // Проекция на ось Y (вертикальная)
    Mat projection = Mat::zeros(1, image.cols, CV_32F);
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            projection.at<float>(0, x) += gray.at<uchar>(y, x);
        }
    }

    // Визуализация
    Mat projectionImage(256, gray.cols, CV_8U, Scalar(255));
    normalize(projection, projection, 0, 255, NORM_MINMAX);
    for (int x = 0; x < gray.cols; x++) {
        line(projectionImage, Point(x, 255), Point(x, 255 - (int)projection.at<float>(0, x)), Scalar(0), 1);
    }
    imshow("Vertical Projection", projectionImage);
    waitKey(0);
    return 0;
}


int imageProfile(Mat& image) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Профиль вдоль центральной строки
    Mat profile = gray.row(gray.rows / 2);
    Mat profileImage(256, profile.cols, CV_8U, Scalar(255));

    for (int x = 0; x < profile.cols; x++) {
        line(profileImage, Point(x, 255), Point(x, 255 - profile.at<uchar>(0, x)), Scalar(0), 1);
    }

    imshow("Image Profile", profileImage);
    waitKey(0);
    return 0;
}


int main() {
    Mat image = imread("C:/image.png"); // Загрузить изображение
    if(image.empty()) {
        cout << "Error loading image" << endl;
        return -1;
    }

    // Разделение на цветовые каналы
    vector<Mat> bgr_planes;
    split(image, bgr_planes);

    // Параметры гистограммы
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    Mat b_hist, g_hist, r_hist;

    // Расчет гистограмм для каждого канала
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange);

    // Выравнивание гистограммы
    equalizeHist(bgr_planes[0], bgr_planes[0]);
    equalizeHist(bgr_planes[1], bgr_planes[1]);
    equalizeHist(bgr_planes[2], bgr_planes[2]);

    // Объединение каналов после выравнивания
    Mat equalized_image;
    merge(bgr_planes, equalized_image);

    imshow("Original Image", image);
    imshow("Equalized Image", equalized_image);

    // Вызов функций для вертикальной проекции и профиля изображения
    verticalProjection(image);
    imageProfile(image);

    waitKey(0);
    return 0;
}
