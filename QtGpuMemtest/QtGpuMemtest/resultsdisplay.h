#ifndef RESULTSDISPLAY_H
#define RESULTSDISPLAY_H

#include <QtGui>
#include <QTextDocument>
#include <QWidget>
#include "ui_resultsdisplay.h"

class ResultsDisplay : public QWidget
{
	Q_OBJECT

public:
	ResultsDisplay(QWidget *parent = 0);
	~ResultsDisplay();

public slots:
	void setResults(QVector<QString>& lines);

private:
	Ui::ResultsDisplay ui;
};

#endif // RESULTSDISPLAY_H
