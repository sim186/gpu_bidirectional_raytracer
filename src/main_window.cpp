#include "main_window.h"
#include <QPainter>
#include <QKeyEvent>
#include <QDebug>

// External globals/functions from main.cu
extern int width;
extern int height;
extern uchar4* pixels;
extern int flag;
extern int all_flag;
extern void update_rendering();
extern void update_rendering_light();
extern void reinit(const int);
extern void save_ppm(int);

MainWindow::MainWindow(QWidget *parent)
    : QWidget(parent)
{
    setFixedSize(::width, ::height);
    setWindowTitle("GPU Bidirectional Raytracer - Qt View");

    m_image = QImage(::width, ::height, QImage::Format_RGB32);
    
    // Timer to trigger rendering updates
    m_timer = new QTimer(this);
    connect(m_timer, &QTimer::timeout, this, &MainWindow::updateRendering);
    m_timer->start(0); // Update as fast as possible, or set a reasonable ms (e.g. 16 for 60fps)
}

MainWindow::~MainWindow()
{
}

void MainWindow::updateRendering()
{
    if (all_flag) {
        // Exit condition handling if needed
        close();
    }

    if (flag == 1)
    {
        update_rendering_light();
    }
    if (flag > 1)
    {
        update_rendering();
    }

    updateImageFromBuffer();
    update(); // Trigger paintEvent
}

void MainWindow::updateImageFromBuffer()
{
    if (!pixels) return;

    // Direct copy might be possible depending on pitch, but simple loop for safety first
    // Pixels is uchar4 (r,g,b,a) usually? OpenGL might be RGBA.
    // QImage::Format_RGB32 expects 0xAARRGGBB.
    // Assuming pixels is RGBA
    
    for (int y = 0; y < ::height; ++y) {
        for (int x = 0; x < ::width; ++x) {
            int idx = y * ::width + x;
            uchar4 p = pixels[idx];
            // Flip Y if needed (OpenGL origin bottom-left vs Qt top-left)?
            // Smallpt usually renders bottom-up or top-down?
            // save_ppm loop: for (int y = height - 1; y >= 0; --y) implies bottom-up buffer.
            // So we invert Y for display.
            
            m_image.setPixel(x, ::height - 1 - y, qRgb(p.x, p.y, p.z));
        }
    }
}

void MainWindow::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    painter.drawImage(0, 0, m_image);
}

void MainWindow::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
    case Qt::Key_Escape:
        close();
        break;
    case Qt::Key_Space:
        reinit(1);
        break;
    case Qt::Key_P:
        save_ppm(0);
        break;
    // Map other keys to camera controls if desired, potentially refactoring key_func from display_functions
    // For now, minimal controls.
    }
}

#include <QApplication>

extern void allocate_buffers();

int run_qt_app(int argc, char* argv[])
{
    QApplication app(argc, argv);
    
    allocate_buffers(); // Allocating CUDA buffers before showing window

    MainWindow window;
    window.show();

    return app.exec();
}
