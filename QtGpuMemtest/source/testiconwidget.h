#ifndef TESTICONWIDGET_H
#define TESTICONWIDGET_H

#include <QtGui>
#include "common.h"

class TestIconWidget : public QWidget
{
	Q_OBJECT

public:
	TestIconWidget(TestInfo& aTestInfo, QWidget *parent = 0);
	~TestIconWidget();

	enum Mode { SelectMode, DisplayMode };

	virtual QSize sizeHint() const;

public slots:
	TestInfo getTestInfo() { return testInfo; };
	void setTestInfo(TestInfo& anotherTestInfo);
	void setTestEnabled(bool x) { testInfo.testEnabled = x; };

	void setStatus(TestStatus aStatus);
	TestStatus getStatus() const { return testStatus; }

	void setState(Mode m);
	Mode getState() const { return widgetMode; }

	void setWidth(int a) { adjWidth = a; }
	int getWidth() const { return adjWidth; }

	void updateRunningColor();

signals:
	void maskEnable(TestInfo pivot, bool mask);

protected:
	virtual void paintEvent(QPaintEvent* event);

	virtual void mousePressEvent(QMouseEvent* event);
	virtual void mouseReleaseEvent(QMouseEvent* event);
	virtual void mouseDoubleClickEvent(QMouseEvent* event);
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
