#include "resultsdisplay.h"

ResultsDisplay::ResultsDisplay(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);
}

ResultsDisplay::~ResultsDisplay()
{

}

void ResultsDisplay::setResults(QVector<QString>& lines)
{
	QString bigString;
	QTextStream bigStream(&bigString);

	for(int i = 0; i < lines.size(); i++)
	{
		bigStream << lines[i] << "\r\n";
	}

	bigStream.flush();
	ui.plainTextEdit->setPlainText(bigString);
}