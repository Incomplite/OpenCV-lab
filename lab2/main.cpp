#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// Функция для отображения изображений
void showImage(const std::string &winName, Mat &img) {
    imshow(winName, img);
    waitKey(0);
    destroyAllWindows();
}

// Функция для эффекта "барреля"
void applyBarrelEffect(const Mat &I) {
    int rows = I.rows;
    int cols = I.cols;

    // Создание сетки
    Mat x_coords = Mat::zeros(rows, cols, CV_32F);
    Mat y_coords = Mat::zeros(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            x_coords.at<float>(y, x) = x;
            y_coords.at<float>(y, x) = y;
        }
    }

    // Нормализация сетки
    float xmid = cols / 2.0;
    float ymid = rows / 2.0;
    x_coords -= xmid;
    y_coords -= ymid;

    // Преобразование в полярные координаты
    Mat r, theta;
    cartToPolar(x_coords / xmid, y_coords / ymid, r, theta);

    // Деформация (баррель-эффект)
    float F3 = 0.1;
    float F5 = 0.12;
    // Преобразование r в Mat для применения pow
    Mat r_pow3 = r.mul(r).mul(r); // r^3
    Mat r_pow5 = r_pow3.mul(r).mul(r); // r^5
    r = r + F3 * r_pow3 + F5 * r_pow5;

    // Обратное преобразование
    Mat u, v;
    polarToCart(r, theta, u, v);

    // Восстановление масштаба
    u = u * xmid + xmid;
    v = v * ymid + ymid;

    // Ремаппинг изображения
    Mat I_barrel;
    remap(I, I_barrel, u, v, INTER_LINEAR);
    showImage("Barrel Effect", I_barrel);
}

// Функция для объединения изображений
void combineImages(const Mat &topPart, const Mat &botPart) {
    int templ_size = 10;

    // Вырезка шаблона
    Mat templ = topPart(Range(topPart.rows - templ_size, topPart.rows), Range::all());

    // Поиск шаблона
    Mat res;
    matchTemplate(botPart, templ, res, TM_CCOEFF);

    // Поиск местоположения максимального значения совпадения
    double min_val, max_val;
    Point min_loc, max_loc;
    minMaxLoc(res, &min_val, &max_val, &min_loc, &max_loc);

    // Изменение размера botPart
    Mat botPart_resized;
    resize(botPart, botPart_resized, Size(topPart.cols, botPart.rows));

    // Создание итогового изображения
    int result_height = topPart.rows + botPart_resized.rows - max_loc.y - templ_size;
    Mat result_img(result_height, topPart.cols, topPart.type(), Scalar(0, 0, 0));

    // Копирование верхней части
    topPart.copyTo(result_img(Range(0, topPart.rows), Range::all()));

    // Копирование нижней части
    Mat bot_crop = botPart_resized(Range(max_loc.y + templ_size, botPart_resized.rows), Range::all());
    bot_crop.copyTo(result_img(Range(topPart.rows, result_img.rows), Range::all()));

    showImage("Combined Image", result_img);
}

int main() {
    Mat img = imread("C:/image.png");
    if (img.empty()) {
        cout << "Ошибка загрузки изображения!" << endl;
        return -1;
    }

    // 1. Сдвиг изображения
    Mat shifted;
    Mat shiftMat = (Mat_<double>(2, 3) << 1, 0, 50, 0, 1, 100);
    warpAffine(img, shifted, shiftMat, img.size());
    showImage("Shifted Image", shifted);

    // 2. Отражение по оси X
    Mat reflected;
    Mat reflectMat = (Mat_<double>(2, 3) << 1, 0, 0, 0, -1, img.rows - 1);
    warpAffine(img, reflected, reflectMat, img.size());
    showImage("Reflected Image", reflected);

    // 3. Масштабирование изображения
    Mat scaled;
    double scaleFactor = 1.5;
    Mat scaleMat = (Mat_<double>(2, 3) << scaleFactor, 0, 0, 0, scaleFactor, 0);
    warpAffine(img, scaled, scaleMat, Size(int(img.cols * scaleFactor), int(img.rows * scaleFactor)));
    showImage("Scaled Image", scaled);

    // 4. Поворот изображения
    Mat rotated;
    double angle = 30.0;
    Point2f center(img.cols / 2.0, img.rows / 2.0);
    Mat rotateMat = getRotationMatrix2D(center, angle, 1);
    warpAffine(img, rotated, rotateMat, img.size());
    showImage("Rotated Image", rotated);

    // 5. Аффинное преобразование
    Mat affineTransformed;
    Point2f srcTri[3] = { Point2f(0, 0), Point2f(img.cols - 1, 0), Point2f(0, img.rows - 1) };
    Point2f dstTri[3] = { Point2f(0, img.rows*0.33), Point2f(img.cols*0.85, img.rows*0.25), Point2f(img.cols*0.15, img.rows*0.7) };
    Mat affineMat = getAffineTransform(srcTri, dstTri);
    warpAffine(img, affineTransformed, affineMat, img.size());
    showImage("Affine Transformed Image", affineTransformed);

    // 6. Проективное преобразование
    Mat projectiveTransformed;
    Point2f srcQuad[4] = { Point2f(0, 0), Point2f(img.cols - 1, 0), Point2f(img.cols - 1, img.rows - 1), Point2f(0, img.rows - 1) };
    Point2f dstQuad[4] = { Point2f(img.cols*0.05, img.rows*0.33), Point2f(img.cols*0.9, img.rows*0.2), Point2f(img.cols*0.8, img.rows*0.9), Point2f(img.cols*0.2, img.rows*0.7) };
    Mat projectiveMat = getPerspectiveTransform(srcQuad, dstQuad);
    warpPerspective(img, projectiveTransformed, projectiveMat, img.size());
    showImage("Projective Transformed Image", projectiveTransformed);

    // --- Дополнительные эффекты ---
    applyBarrelEffect(img);

    Mat topPart = imread("C:/cat.jpg");
    Mat botPart = imread("C:/roadSign.jpg");
    if (!topPart.empty() && !botPart.empty()) {
        combineImages(topPart, botPart);
    }

    return 0;
}
