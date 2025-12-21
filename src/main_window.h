#pragma once

#include <QWidget>
#include <QImage>
#include <QTimer>
#include <QLabel>
#include <vector_types.h> // For uchar4

class MainWindow : public QWidget
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void paintEvent(QPaintEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private slots:
    void updateRendering();

private:
    QImage m_image;
    QTimer *m_timer;
    
    // Helper to Convert uchar4 buffer to QImage
    void updateImageFromBuffer();
};
