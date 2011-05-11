#ifndef TESTICONWIDGET_H
#define TESTICONWIDGET_H

#include <QtGui>
#include "gputests.h"

enum TestStatus { TestNotStarted, TestPassed, TestFailed, TestRunning };

class TestIconWidget : public QWidget
{
	Q_OBJECT

public:
	TestIconWidget(TestInfo& aTestInfo, QWidget *parent = 0);
	~TestIconWidget();

	static enum Mode { SelectMode, DisplayMode };

	virtual QSize sizeHint() const;

public slots:
	void setStatus(TestStatus aStatus);
	TestStatus getStatus() const { return testStatus; }

	//void setTestInfo();
	TestInfo getTestInfo() { return testInfo; }

	void setMode(Mode m);
	Mode getMode() const { return widgetMode; }

	void setWidth(int a) { adjWidth = a; }
	int getWidth() const { return adjWidth; }

	void updateRunningColor();

protected:
	virtual void paintEvent(QPaintEvent* event);

	virtual void mousePressEvent(QMouseEvent* event);
	virtual void mouseReleaseEvent(QMouseEvent* event);
	virtual void enterEvent(QEvent* event);
	virtual void leaveEvent(QEvent* event);

private:
	TestStatus	testStatus;
	TestInfo	testInfo;
	Mode		widgetMode;

	QTimer*		flashTimer;
	QColor		flashColor;
	bool		mouseClicked;
	bool		hover;

	int			adjWidth;

};

#endif