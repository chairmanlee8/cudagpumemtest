/********************************************************************************
** Form generated from reading UI file 'resultsdisplay.ui'
**
** Created: Wed Apr 27 14:33:54 2011
**      by: Qt User Interface Compiler version 4.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_RESULTSDISPLAY_H
#define UI_RESULTSDISPLAY_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QPlainTextEdit>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ResultsDisplay
{
public:
    QHBoxLayout *horizontalLayout;
    QPlainTextEdit *plainTextEdit;

    void setupUi(QWidget *ResultsDisplay)
    {
        if (ResultsDisplay->objectName().isEmpty())
            ResultsDisplay->setObjectName(QString::fromUtf8("ResultsDisplay"));
        ResultsDisplay->resize(580, 652);
        horizontalLayout = new QHBoxLayout(ResultsDisplay);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        plainTextEdit = new QPlainTextEdit(ResultsDisplay);
        plainTextEdit->setObjectName(QString::fromUtf8("plainTextEdit"));
        QFont font;
        font.setFamily(QString::fromUtf8("Courier New"));
        font.setPointSize(10);
        plainTextEdit->setFont(font);
        plainTextEdit->setLineWrapMode(QPlainTextEdit::NoWrap);
        plainTextEdit->setReadOnly(true);

        horizontalLayout->addWidget(plainTextEdit);


        retranslateUi(ResultsDisplay);

        QMetaObject::connectSlotsByName(ResultsDisplay);
    } // setupUi

    void retranslateUi(QWidget *ResultsDisplay)
    {
        ResultsDisplay->setWindowTitle(QApplication::translate("ResultsDisplay", "Results Display", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ResultsDisplay: public Ui_ResultsDisplay {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_RESULTSDISPLAY_H
