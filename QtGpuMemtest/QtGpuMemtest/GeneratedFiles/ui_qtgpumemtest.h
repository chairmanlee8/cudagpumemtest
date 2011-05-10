/********************************************************************************
** Form generated from reading UI file 'qtgpumemtest.ui'
**
** Created: Tue May 3 13:58:14 2011
**      by: Qt User Interface Compiler version 4.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QTGPUMEMTEST_H
#define UI_QTGPUMEMTEST_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCommandLinkButton>
#include <QtGui/QDial>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QProgressBar>
#include <QtGui/QSpacerItem>
#include <QtGui/QStackedWidget>
#include <QtGui/QToolBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QtGpuMemtestClass
{
public:
    QAction *actionExit;
    QAction *actionAbout;
    QAction *actionGuide;
    QAction *actionRelist;
    QAction *actionStartChecked;
    QAction *actionCheckAll;
    QAction *actionStopAll;
    QAction *actionCheckNone;
    QAction *actionExportResults;
    QAction *actionClipboardResults;
    QAction *actionMonitor_temperatures;
    QAction *actionMonitor_fan_speeds;
    QAction *actionAlways_start_in_advanced_mode;
    QAction *actionShowWizardOnStartup;
    QAction *actionMonitor_temperatures_2;
    QAction *actionMonitor_fan_speeds_2;
    QAction *actionSwitchView;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout_2;
    QStackedWidget *stackedWidget;
    QWidget *pageBasic;
    QVBoxLayout *verticalLayout_3;
    QCommandLinkButton *quickTestButton;
    QHBoxLayout *horizontalLayout;
    QCommandLinkButton *customStressTestButton;
    QDial *customStressDial;
    QSpacerItem *verticalSpacer;
    QVBoxLayout *verticalLayout;
    QLabel *labelGpuTemperatures;
    QLabel *labelFanSpeeds;
    QProgressBar *progressBarOverall;
    QWidget *pageAdvanced;
    QVBoxLayout *verticalLayout_5;
    QVBoxLayout *verticalLayoutGpus;
    QWidget *pageResults;
    QVBoxLayout *verticalLayout_6;
    QLabel *labelPassFail;
    QCommandLinkButton *buttonReturn;
    QWidget *pageNoDevices;
    QVBoxLayout *verticalLayout_4;
    QLabel *labelNoDevices;
    QMenuBar *menuBar;
    QMenu *menu_File;
    QMenu *menu_Help;
    QMenu *menu_View;
    QToolBar *mainToolBar;

    void setupUi(QMainWindow *QtGpuMemtestClass)
    {
        if (QtGpuMemtestClass->objectName().isEmpty())
            QtGpuMemtestClass->setObjectName(QString::fromUtf8("QtGpuMemtestClass"));
        QtGpuMemtestClass->resize(429, 511);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/QtGpuMemtest/ncsa.ico"), QSize(), QIcon::Normal, QIcon::Off);
        QtGpuMemtestClass->setWindowIcon(icon);
        actionExit = new QAction(QtGpuMemtestClass);
        actionExit->setObjectName(QString::fromUtf8("actionExit"));
        actionAbout = new QAction(QtGpuMemtestClass);
        actionAbout->setObjectName(QString::fromUtf8("actionAbout"));
        actionGuide = new QAction(QtGpuMemtestClass);
        actionGuide->setObjectName(QString::fromUtf8("actionGuide"));
        actionRelist = new QAction(QtGpuMemtestClass);
        actionRelist->setObjectName(QString::fromUtf8("actionRelist"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/QtGpuMemtest/loop_24x24.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionRelist->setIcon(icon1);
        actionStartChecked = new QAction(QtGpuMemtestClass);
        actionStartChecked->setObjectName(QString::fromUtf8("actionStartChecked"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/QtGpuMemtest/play_18x24.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionStartChecked->setIcon(icon2);
        actionCheckAll = new QAction(QtGpuMemtestClass);
        actionCheckAll->setObjectName(QString::fromUtf8("actionCheckAll"));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/QtGpuMemtest/check_24x20.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCheckAll->setIcon(icon3);
        actionCheckAll->setVisible(true);
        actionStopAll = new QAction(QtGpuMemtestClass);
        actionStopAll->setObjectName(QString::fromUtf8("actionStopAll"));
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/QtGpuMemtest/stop_16x16.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionStopAll->setIcon(icon4);
        actionCheckNone = new QAction(QtGpuMemtestClass);
        actionCheckNone->setObjectName(QString::fromUtf8("actionCheckNone"));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/QtGpuMemtest/x_21x21.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCheckNone->setIcon(icon5);
        actionExportResults = new QAction(QtGpuMemtestClass);
        actionExportResults->setObjectName(QString::fromUtf8("actionExportResults"));
        actionClipboardResults = new QAction(QtGpuMemtestClass);
        actionClipboardResults->setObjectName(QString::fromUtf8("actionClipboardResults"));
        actionMonitor_temperatures = new QAction(QtGpuMemtestClass);
        actionMonitor_temperatures->setObjectName(QString::fromUtf8("actionMonitor_temperatures"));
        actionMonitor_temperatures->setCheckable(true);
        actionMonitor_temperatures->setChecked(true);
        actionMonitor_fan_speeds = new QAction(QtGpuMemtestClass);
        actionMonitor_fan_speeds->setObjectName(QString::fromUtf8("actionMonitor_fan_speeds"));
        actionMonitor_fan_speeds->setCheckable(true);
        actionMonitor_fan_speeds->setChecked(true);
        actionAlways_start_in_advanced_mode = new QAction(QtGpuMemtestClass);
        actionAlways_start_in_advanced_mode->setObjectName(QString::fromUtf8("actionAlways_start_in_advanced_mode"));
        actionAlways_start_in_advanced_mode->setCheckable(true);
        actionShowWizardOnStartup = new QAction(QtGpuMemtestClass);
        actionShowWizardOnStartup->setObjectName(QString::fromUtf8("actionShowWizardOnStartup"));
        actionShowWizardOnStartup->setCheckable(true);
        actionShowWizardOnStartup->setChecked(true);
        actionMonitor_temperatures_2 = new QAction(QtGpuMemtestClass);
        actionMonitor_temperatures_2->setObjectName(QString::fromUtf8("actionMonitor_temperatures_2"));
        actionMonitor_temperatures_2->setCheckable(true);
        actionMonitor_temperatures_2->setChecked(true);
        actionMonitor_fan_speeds_2 = new QAction(QtGpuMemtestClass);
        actionMonitor_fan_speeds_2->setObjectName(QString::fromUtf8("actionMonitor_fan_speeds_2"));
        actionMonitor_fan_speeds_2->setCheckable(true);
        actionMonitor_fan_speeds_2->setChecked(true);
        actionSwitchView = new QAction(QtGpuMemtestClass);
        actionSwitchView->setObjectName(QString::fromUtf8("actionSwitchView"));
        QIcon icon6;
        icon6.addFile(QString::fromUtf8(":/QtGpuMemtest/magnifying_glass_alt_24x24.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSwitchView->setIcon(icon6);
        centralWidget = new QWidget(QtGpuMemtestClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        verticalLayout_2 = new QVBoxLayout(centralWidget);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        stackedWidget = new QStackedWidget(centralWidget);
        stackedWidget->setObjectName(QString::fromUtf8("stackedWidget"));
        stackedWidget->setFrameShape(QFrame::NoFrame);
        pageBasic = new QWidget();
        pageBasic->setObjectName(QString::fromUtf8("pageBasic"));
        verticalLayout_3 = new QVBoxLayout(pageBasic);
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        quickTestButton = new QCommandLinkButton(pageBasic);
        quickTestButton->setObjectName(QString::fromUtf8("quickTestButton"));

        verticalLayout_3->addWidget(quickTestButton);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setSizeConstraint(QLayout::SetDefaultConstraint);
        customStressTestButton = new QCommandLinkButton(pageBasic);
        customStressTestButton->setObjectName(QString::fromUtf8("customStressTestButton"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(customStressTestButton->sizePolicy().hasHeightForWidth());
        customStressTestButton->setSizePolicy(sizePolicy);

        horizontalLayout->addWidget(customStressTestButton);

        customStressDial = new QDial(pageBasic);
        customStressDial->setObjectName(QString::fromUtf8("customStressDial"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(customStressDial->sizePolicy().hasHeightForWidth());
        customStressDial->setSizePolicy(sizePolicy1);
        customStressDial->setMinimum(30);
        customStressDial->setMaximum(360);
        customStressDial->setSingleStep(10);
        customStressDial->setPageStep(30);
        customStressDial->setValue(30);
        customStressDial->setOrientation(Qt::Horizontal);
        customStressDial->setNotchTarget(10);
        customStressDial->setNotchesVisible(true);

        horizontalLayout->addWidget(customStressDial);


        verticalLayout_3->addLayout(horizontalLayout);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        labelGpuTemperatures = new QLabel(pageBasic);
        labelGpuTemperatures->setObjectName(QString::fromUtf8("labelGpuTemperatures"));

        verticalLayout->addWidget(labelGpuTemperatures);

        labelFanSpeeds = new QLabel(pageBasic);
        labelFanSpeeds->setObjectName(QString::fromUtf8("labelFanSpeeds"));

        verticalLayout->addWidget(labelFanSpeeds);

        progressBarOverall = new QProgressBar(pageBasic);
        progressBarOverall->setObjectName(QString::fromUtf8("progressBarOverall"));
        progressBarOverall->setEnabled(false);
        progressBarOverall->setValue(0);

        verticalLayout->addWidget(progressBarOverall);


        verticalLayout_3->addLayout(verticalLayout);

        stackedWidget->addWidget(pageBasic);
        pageAdvanced = new QWidget();
        pageAdvanced->setObjectName(QString::fromUtf8("pageAdvanced"));
        verticalLayout_5 = new QVBoxLayout(pageAdvanced);
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setContentsMargins(11, 11, 11, 11);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        verticalLayoutGpus = new QVBoxLayout();
        verticalLayoutGpus->setSpacing(6);
        verticalLayoutGpus->setObjectName(QString::fromUtf8("verticalLayoutGpus"));

        verticalLayout_5->addLayout(verticalLayoutGpus);

        stackedWidget->addWidget(pageAdvanced);
        pageResults = new QWidget();
        pageResults->setObjectName(QString::fromUtf8("pageResults"));
        verticalLayout_6 = new QVBoxLayout(pageResults);
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setContentsMargins(11, 11, 11, 11);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        labelPassFail = new QLabel(pageResults);
        labelPassFail->setObjectName(QString::fromUtf8("labelPassFail"));
        QFont font;
        font.setPointSize(14);
        labelPassFail->setFont(font);
        labelPassFail->setAlignment(Qt::AlignCenter);

        verticalLayout_6->addWidget(labelPassFail);

        buttonReturn = new QCommandLinkButton(pageResults);
        buttonReturn->setObjectName(QString::fromUtf8("buttonReturn"));
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(buttonReturn->sizePolicy().hasHeightForWidth());
        buttonReturn->setSizePolicy(sizePolicy2);

        verticalLayout_6->addWidget(buttonReturn);

        stackedWidget->addWidget(pageResults);
        pageNoDevices = new QWidget();
        pageNoDevices->setObjectName(QString::fromUtf8("pageNoDevices"));
        verticalLayout_4 = new QVBoxLayout(pageNoDevices);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        labelNoDevices = new QLabel(pageNoDevices);
        labelNoDevices->setObjectName(QString::fromUtf8("labelNoDevices"));
        labelNoDevices->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);
        labelNoDevices->setWordWrap(true);

        verticalLayout_4->addWidget(labelNoDevices);

        stackedWidget->addWidget(pageNoDevices);

        verticalLayout_2->addWidget(stackedWidget);

        QtGpuMemtestClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(QtGpuMemtestClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 429, 21));
        menu_File = new QMenu(menuBar);
        menu_File->setObjectName(QString::fromUtf8("menu_File"));
        menu_Help = new QMenu(menuBar);
        menu_Help->setObjectName(QString::fromUtf8("menu_Help"));
        menu_View = new QMenu(menuBar);
        menu_View->setObjectName(QString::fromUtf8("menu_View"));
        QtGpuMemtestClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(QtGpuMemtestClass);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        mainToolBar->setEnabled(true);
        mainToolBar->setMovable(false);
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        mainToolBar->setFloatable(false);
        QtGpuMemtestClass->addToolBar(Qt::TopToolBarArea, mainToolBar);

        menuBar->addAction(menu_File->menuAction());
        menuBar->addAction(menu_View->menuAction());
        menuBar->addAction(menu_Help->menuAction());
        menu_File->addAction(actionShowWizardOnStartup);
        menu_File->addSeparator();
        menu_File->addAction(actionClipboardResults);
        menu_File->addAction(actionExportResults);
        menu_File->addSeparator();
        menu_File->addAction(actionExit);
        menu_Help->addAction(actionGuide);
        menu_Help->addSeparator();
        menu_Help->addAction(actionAbout);
        menu_View->addAction(actionMonitor_temperatures_2);
        menu_View->addAction(actionMonitor_fan_speeds_2);
        mainToolBar->addAction(actionRelist);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionSwitchView);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionCheckAll);
        mainToolBar->addAction(actionCheckNone);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionStartChecked);
        mainToolBar->addAction(actionStopAll);

        retranslateUi(QtGpuMemtestClass);

        stackedWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(QtGpuMemtestClass);
    } // setupUi

    void retranslateUi(QMainWindow *QtGpuMemtestClass)
    {
        QtGpuMemtestClass->setWindowTitle(QApplication::translate("QtGpuMemtestClass", "CUDA GPU Memtest", 0, QApplication::UnicodeUTF8));
        actionExit->setText(QApplication::translate("QtGpuMemtestClass", "Exit", 0, QApplication::UnicodeUTF8));
        actionAbout->setText(QApplication::translate("QtGpuMemtestClass", "About", 0, QApplication::UnicodeUTF8));
        actionGuide->setText(QApplication::translate("QtGpuMemtestClass", "Guide...", 0, QApplication::UnicodeUTF8));
        actionRelist->setText(QApplication::translate("QtGpuMemtestClass", "Relist", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionRelist->setToolTip(QApplication::translate("QtGpuMemtestClass", "Relist devices.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        actionStartChecked->setText(QApplication::translate("QtGpuMemtestClass", "Start Checked", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionStartChecked->setToolTip(QApplication::translate("QtGpuMemtestClass", "Start all checked GPUs.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        actionCheckAll->setText(QApplication::translate("QtGpuMemtestClass", "Check All", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionCheckAll->setToolTip(QApplication::translate("QtGpuMemtestClass", "Check all devices.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        actionStopAll->setText(QApplication::translate("QtGpuMemtestClass", "Stop All", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionStopAll->setToolTip(QApplication::translate("QtGpuMemtestClass", "Stop all tests on all devices.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        actionCheckNone->setText(QApplication::translate("QtGpuMemtestClass", "Check None", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionCheckNone->setToolTip(QApplication::translate("QtGpuMemtestClass", "Uncheck all devices.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        actionExportResults->setText(QApplication::translate("QtGpuMemtestClass", "Export results...", 0, QApplication::UnicodeUTF8));
        actionClipboardResults->setText(QApplication::translate("QtGpuMemtestClass", "Copy results to clipboard", 0, QApplication::UnicodeUTF8));
        actionMonitor_temperatures->setText(QApplication::translate("QtGpuMemtestClass", "Monitor temperatures", 0, QApplication::UnicodeUTF8));
        actionMonitor_fan_speeds->setText(QApplication::translate("QtGpuMemtestClass", "Monitor fan speeds", 0, QApplication::UnicodeUTF8));
        actionAlways_start_in_advanced_mode->setText(QApplication::translate("QtGpuMemtestClass", "Always start in advanced mode", 0, QApplication::UnicodeUTF8));
        actionShowWizardOnStartup->setText(QApplication::translate("QtGpuMemtestClass", "Show wizard on startup", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionShowWizardOnStartup->setToolTip(QApplication::translate("QtGpuMemtestClass", "Show wizard on startup. If not checked, the application will default to the advanced view.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        actionMonitor_temperatures_2->setText(QApplication::translate("QtGpuMemtestClass", "Monitor temperatures", 0, QApplication::UnicodeUTF8));
        actionMonitor_fan_speeds_2->setText(QApplication::translate("QtGpuMemtestClass", "Monitor fan speeds", 0, QApplication::UnicodeUTF8));
        actionSwitchView->setText(QApplication::translate("QtGpuMemtestClass", "Advanced View", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionSwitchView->setToolTip(QApplication::translate("QtGpuMemtestClass", "Switch to advanced view.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        quickTestButton->setText(QApplication::translate("QtGpuMemtestClass", "Quick Test", 0, QApplication::UnicodeUTF8));
        quickTestButton->setDescription(QApplication::translate("QtGpuMemtestClass", "This fast diagnostic will scan your graphics cards for any obvious memory faults.", 0, QApplication::UnicodeUTF8));
        customStressTestButton->setText(QApplication::translate("QtGpuMemtestClass", "30 Minute Stress Burn", 0, QApplication::UnicodeUTF8));
        customStressTestButton->setDescription(QApplication::translate("QtGpuMemtestClass", "This intensive test will push your graphics cards to their limit.", 0, QApplication::UnicodeUTF8));
        labelGpuTemperatures->setText(QApplication::translate("QtGpuMemtestClass", "GPU Temperatures:", 0, QApplication::UnicodeUTF8));
        labelFanSpeeds->setText(QApplication::translate("QtGpuMemtestClass", "GPU Fan Speeds:", 0, QApplication::UnicodeUTF8));
        progressBarOverall->setFormat(QApplication::translate("QtGpuMemtestClass", " Test Progress: %p%", 0, QApplication::UnicodeUTF8));
        labelPassFail->setText(QApplication::translate("QtGpuMemtestClass", "Passed", 0, QApplication::UnicodeUTF8));
        buttonReturn->setText(QApplication::translate("QtGpuMemtestClass", "Return Home", 0, QApplication::UnicodeUTF8));
        buttonReturn->setDescription(QApplication::translate("QtGpuMemtestClass", "Go back to the testing screen for another test.", 0, QApplication::UnicodeUTF8));
        labelNoDevices->setText(QApplication::translate("QtGpuMemtestClass", "We're sorry, we couldn't find any CUDA enabled devices on your computer. Try installing CUDA-enabled drivers then restarting this program.", 0, QApplication::UnicodeUTF8));
        menu_File->setTitle(QApplication::translate("QtGpuMemtestClass", "&File", 0, QApplication::UnicodeUTF8));
        menu_Help->setTitle(QApplication::translate("QtGpuMemtestClass", "&Help", 0, QApplication::UnicodeUTF8));
        menu_View->setTitle(QApplication::translate("QtGpuMemtestClass", "&View", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class QtGpuMemtestClass: public Ui_QtGpuMemtestClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTGPUMEMTEST_H
